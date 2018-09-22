import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import viz
import utils
from explainers_redo import zero_grad
from explainers_redo import SparseExplainer, RobustSparseExplainer

import matplotlib.pyplot as plt
from PIL import Image
from plotnine import ggplot, aes, geom_density, facet_grid


transf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def get_topk_mask(saliency, k=1e4, topk_agg=None, flip=False):
    '''Generate binary mask based on saliency value.
    Saliency is first aggregated via `topk_agg`, and the mask has the same
    shape as the aggregated saliency (3D or 4D).
    If `flip` is False, high saliency value corresponds to 1, otherwise 0.
    :param saliency: 4D numpy.array
    :param k: number of pixels to attack, channel independent if mask is 4D
    :param topk_agg: should probably move to whoever calls me
    :param flip: if flip, high value is 0 instead of 1
    '''
    assert len(saliency.shape) == 4
    batch_size, n_chs, height, width = saliency.shape
    if topk_agg is not None:
        saliency = topk_agg(saliency)

    if len(saliency.shape) == 4:
        saliency = saliency.reshape(batch_size, n_chs, -1)
        topk_mask = np.ones_like(saliency) if flip else np.zeros_like(saliency)
        topk_idx = np.argsort(-saliency, axis=2)[:, :, :k]
        for i in range(batch_size):
            for j in range(n_chs):
                topk_mask[i, j, topk_idx[i, j]] = 0 if flip else 1
        topk_mask = topk_mask.reshape(batch_size, n_chs, height, width)
    elif len(saliency.shape) == 3:
        saliency = saliency.reshape(batch_size, -1)
        topk_mask = np.ones_like(saliency) if flip else np.zeros_like(saliency)
        topk_idx = np.argsort(-saliency, axis=1)[:, :k]
        for i in range(batch_size):
            topk_mask[i, topk_idx[i]] = 0 if flip else 1
        topk_mask = topk_mask.reshape(batch_size, height, width)
        topk_mask = np.expand_dims(topk_mask, axis=1)
        topk_mask = np.tile(topk_mask, (1, n_chs, 1, 1))
    else:
        print('saliency shape wrong')
    return topk_mask


class GhorbaniAttack:

    def __init__(self,
                 lambda_t1=0,
                 lambda_t2=1,
                 lambda_l1=0,
                 lambda_l2=0,
                 n_iter=20,
                 optim='sgd',
                 lr=1e-2,
                 epsilon=2/255,
                 k=1e4,
                 topk_agg=lambda x: np.abs(x)
                 ):
        '''
        :param lambda_t1
        :param lambda_t2
        :param lambda_l1
        :param lambda_l2
        :param n_iter
        :param optim
        :param lr
        :param epsilon
        :param k: number of pixels to dampen
        :param topk_agg: aggregation for selecting top-k pixels
        '''
        self.lambda_t1 = lambda_t1
        self.lambda_t2 = lambda_t2
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.n_iter = n_iter
        self.optim = optim.lower()
        self.lr = lr
        self.epsilon = epsilon
        self.k = int(k)
        self.topk_agg = topk_agg

    def get_input_grad(self, x, output, y, create_graph=False):
        '''two methods for getting input gradient'''
        loss = F.cross_entropy(output, y)
        x_grad, = torch.autograd.grad(loss, x, create_graph=create_graph)

        # grad_out = torch.zeros_like(output.data)
        # grad_out.scatter_(1, y.data.unsqueeze(0).t(), 1.0)
        # x_grad, = torch.autograd.grad(output, x,
        #                               grad_outputs=grad_out,
        #                               create_graph=create_graph)

        return x_grad

    def attack(self, model, x, saliency=None):
        '''Generate attack against specified saliency mapping.
        If saliency is not specified, assume vanila gradient saliency.
        '''
        batch_size, n_chs, height, width = x.shape

        x_prev = x.clone()
        delta = torch.zeros_like(x)
        y_org = model(Variable(x)).max(1)[1].data

        if saliency is None:
            x_curr = Variable(x, requires_grad=True)
            output = model(x_curr)
            y = output.max(1)[1]
            saliency = self.get_input_grad(x_curr, output, y)
            saliency = saliency.data.cpu().numpy()

        topk_mask = get_topk_mask(saliency, self.k, topk_agg=self.topk_agg)
        topk_mask = torch.FloatTensor(topk_mask).cuda()
        topk_mask = Variable(topk_mask)

        step_size = self.epsilon / self.n_iter
        stopped = [False for _ in range(batch_size)]
        for i in range(self.n_iter):
            zero_grad(model)
            x_curr = Variable(x, requires_grad=True)
            output = model(x_curr)
            y = output.max(1)[1]

            x_grad = self.get_input_grad(x_curr, output, y, create_graph=True)

            topk = (x_grad.abs() * topk_mask).sum()
            delta, = torch.autograd.grad(-topk, x_curr)
            delta = delta.sign().data

            # verify same prediction
            for bidx in range(batch_size):
                if stopped[bidx] or y.data[bidx] != y_org[bidx]:
                    x[bidx] = x_prev[bidx]
                    delta[bidx].zero_()
                    stopped[bidx] = True

            if all(stopped):
                break

            x_prev = x.clone()
            x = torch.clamp(x + step_size * delta, 0, 1)

        # final check that all predictions remain
        y = model(Variable(x)).max(1)[1].data
        assert (y == y_org).all()
        return x


class ScaledNoiseAttack:

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def attack(self, model, x, saliency=None):
        x = x.cpu().numpy()
        noise = 2 * np.random.randint(2, size=x.shape) - 1
        noise = np.sign(noise) * self.epsilon
        x = np.clip(x + noise * x, 0, 1)
        x = torch.FloatTensor(x)
        return x


class FGSM:

    def __init__(self, epsilon=2 / 255, n_iter=10):
        self.epsilon = epsilon
        self.n_iter = n_iter

    def get_input_grad(self, x, output, y, create_graph=False):
        '''two methods for getting input gradient'''
        loss = F.cross_entropy(output, y)
        x_grad, = torch.autograd.grad(loss, x, create_graph=create_graph)

        # grad_out = torch.zeros_like(output.data)
        # grad_out.scatter_(1, y.data.unsqueeze(0).t(), 1.0)
        # x_grad, = torch.autograd.grad(output, x,
        #                               grad_outputs=grad_out,
        #                               create_graph=create_graph)
        return x_grad

    def attack(self, model, x, saliency=None):
        batch_size, n_chs, height, width = x.shape
        step_size = self.epsilon / self.n_iter
        for i in range(self.n_iter):
            zero_grad(model)
            x_curr = Variable(x, requires_grad=True)
            output = model(x_curr)
            y = output.max(1)[1]
            x_grad = self.get_input_grad(x, output, y).data
            x_grad = x_grad.sign()
            x = torch.clamp(x + step_size * x_grad, 0, 1)
        return x


def aggregate(saliency):
    '''combine saliency mapping with image
    from 4D (bsz, 3, h, w) to 3D (bsz, h, w)
    '''
    return np.abs(saliency).sum(dim=1)


def saliency_correlation(s1, s2, image):
    # s1 and s2 are batched
    s1 = aggregate(s1, image)
    s2 = aggregate(s2, image)
    assert s1.shape == s2.shape
    assert s1.ndimension() == 3  # batch, height, width
    batch_size = s1.shape[0]
    s1 = s1.reshape(batch_size, -1)
    s2 = s2.reshape(batch_size, -1)
    scores = []
    for x1, x2 in zip(s1, s2):
        scores.append(spearmanr(x1, x2).correlation)
    return scores


def channel_correlation(s1, s2, image):
    assert s1.shape == s2.shape
    assert s1.ndimension() == 4  # batch, 3, height, width
    batch_size = s1.shape[0]
    s1 = np.abs(s1).reshape(batch_size, 3, -1)
    s2 = np.abs(s2).reshape(batch_size, 3, -1)
    scores = []
    for x1, x2 in zip(s1, s2):
        scores.append(spearmanr(x1[k], x2[k]).correlation for k in range(3))
    scores = list(map(list, zip(*scores)))
    return scores


def attack_batch(model, batch, explainers, attackers, return_saliency=False):
    scores = []
    saliency_maps = []
    batch_size = batch.shape[0]
    for mth_name, explainer in explainers:
        saliency_1 = explainer.explain(model, batch.clone()).cpu().numpy()
        for atk_name, attacker in attackers:
            perturbed = attacker.attack(batch.clone(), saliency_1)
            perturbed_np = perturbed.cpu().numpy()
            perturbed = Variable(perturbed, requires_grad=True)
            saliency_2 = explainer.explain(model, batch.clone()).cpu().numpy()

            ss = [
                saliency_correlation(saliency_1, saliency_2),
                *channel_correlation(saliency_1, saliency_2),
            ]
            ss = list(map(list, zip(*ss)))

            for i in range(batch_size):
                meta_data = [i, mth_name, atk_name]
                scores.append(meta_data + ss[i])
                if return_saliency:
                    saliency_maps.append(
                        meta_data +
                        [saliency_1[i], saliency_2[i], perturbed_np[i]]
                    )
    if return_saliency:
        return scores, saliency_maps
    else:
        return scores


def run_attack(model, batches, explainers, attackers):
    results = []
    for batch_idx, batch in enumerate(batches):
        # TODO
        # labels = [y for x, y in batch]
        # batch = torch.stack([transf(x) for x, y in batch]).cuda()
        batch = torch.stack([transf(x) for x in batch]).cuda()
        results += attack_batch(model, batch, explainers, attackers)
    n_scores = len(results[0]) - 2  # number of different scores
    columns = (
        ['method', 'attack'] +
        ['score_{}'.format(i) for i in range(n_scores)]
    )
    df = pd.DataFrame(results, columns=columns)
    print(df.groupby(['attack', 'method']).mean())


def setup_imagenet(batch_size=16, n_batches=-1, n_images=-1):
    model = utils.load_model('resnet50')
    model.eval()
    model.cuda()
    print('model loaded')

    def batch_loader(image_files):
        return [viz.pil_loader(x) for x in image_files]

    # TODO also load labels
    # TODO add indices
    image_path = '/fs/imageNet/imagenet/ILSVRC_val/**/*.JPEG'
    image_files = list(glob.iglob(image_path, recursive=True))
    np.random.seed(0)
    np.random.shuffle(image_files)

    if n_images > 0:
        image_files = image_files[:n_images]
    elif n_batches > 0:
        image_files = image_files[:batch_size * n_batches]

    batch_indices = list(range(0, len(image_files), batch_size))
    batch_files = [image_files[i: i + batch_size] for i in batch_indices]
    batches = map(batch_loader, batch_files)
    print('image loaded', len(batch_files))

    return model, batches


if __name__ == '__main__':
    epsilon = 2 / 255

    explainers = [
        ('Sparse (l1=0.5)', SparseExplainer(lambda_l1=0.5)),
    ]

    attackers = [
        ('Ghorbani',
         GhorbaniAttack(
             epsilon=epsilon,
             topk_agg=lambda x: np.abs(x),
             # topk_agg=lambda x: np.abs(x).sum(1),
         )),
        ('Random', ScaledNoiseAttack(epsilon=epsilon)),
    ]

    model, batches = setup_imagenet(n_batches=1)
    run_attack(model, batches, explainers, attackers)
