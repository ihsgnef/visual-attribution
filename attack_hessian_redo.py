import os
import json
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
from explainers_redo import SparseExplainer, RobustSparseExplainer, \
    VanillaGradExplainer, IntegrateGradExplainer, SmoothGradExplainer

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


class EmptyAttack:

    def attack(self, model, x, saliency=None):
        return x


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
            for idx in range(batch_size):
                if stopped[idx] or y.data[idx] != y_org[idx]:
                    x[idx] = x_prev[idx]
                    delta[idx].zero_()
                    stopped[idx] = True

            if all(stopped):
                break

            x_prev = x.clone()
            x = torch.clamp(x + step_size * delta, 0, 1)

        # final check that all predictions remain
        y = model(Variable(x)).max(1)[1].data
        assert (y == y_org).all()
        return x


class ScaledNoiseAttack:

    def __init__(self, epsilon=2 / 255):
        self.epsilon = epsilon

    def attack(self, model, x, saliency=None):
        x = x.cpu().numpy()
        noise = 2 * np.random.randint(2, size=x.shape) - 1
        noise = np.sign(noise) * self.epsilon
        x = np.clip(x + noise * x, 0, 1)
        x = torch.FloatTensor(x).cuda()
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


def saliency_correlation(s1, s2):
    # s1 and s2 are batched
    assert s1.shape == s2.shape
    assert s1.ndim == 4
    batch_size = s1.shape[0]
    s1 = np.abs(s1).sum(axis=1)
    s2 = np.abs(s2).sum(axis=1)
    s1 = s1.reshape(batch_size, -1)
    s2 = s2.reshape(batch_size, -1)
    return [spearmanr(x1, x2).correlation for x1, x2 in zip(s1, s2)]


def channel_correlation(s1, s2):
    assert s1.shape == s2.shape
    assert s1.ndim == 4  # batch, 3, height, width
    batch_size = s1.shape[0]
    s1 = np.abs(s1).reshape(batch_size, 3, -1)
    s2 = np.abs(s2).reshape(batch_size, 3, -1)
    scores = []
    for x1, x2 in zip(s1, s2):
        scores.append(spearmanr(x1[k], x2[k]).correlation for k in range(3))
    scores = list(map(list, zip(*scores)))
    return scores


def attack_batch(model, batch, explainers, attackers, return_saliency=False):
    results = []
    batch_size = batch.shape[0]
    for mth_name, explainer in explainers:
        saliency_1 = explainer.explain(model, batch.clone()).cpu().numpy()
        for atk_name, attacker in attackers:
            perturbed = attacker.attack(model, batch.clone(), saliency_1)
            perturbed_np = perturbed.cpu().numpy()
            saliency_2 = explainer.explain(model, perturbed).cpu().numpy()

            # scores = [
            #     saliency_correlation(saliency_1, saliency_2),
            #     *channel_correlation(saliency_1, saliency_2),
            # ]
            scores = saliency_correlation(saliency_1, saliency_2)

            for i in range(batch_size):
                row = {
                    'idx': i,
                    'explainer': mth_name,
                    'attacker': atk_name,
                    'spearman': scores[i],
                }
                if return_saliency:
                    row.update({
                        'saliency_1': saliency_1[i],
                        'saliency_2': saliency_2[i],
                        'perturbed': perturbed_np[i],
                    })
                results.append(row)
    return results


def setup_imagenet(batch_size=16, n_batches=-1, n_examples=-1):
    model = utils.load_model('resnet50')
    model.eval()
    model.cuda()
    print('model loaded')

    image_path = '/fs/imageNet/imagenet/ILSVRC_val/**/*.JPEG'
    image_files = list(glob.iglob(image_path, recursive=True))
    image_files = sorted(image_files, key=lambda x: os.path.basename(x))

    from imagenet1000_clsid_to_human import clsid_to_human

    label_path = '/fs/imageNet/imagenet/ILSVRC2012_devkit_t12/' \
                 + 'data/ILSVRC2012_validation_ground_truth.txt'
    with open(label_path) as f:
        labels = [clsid_to_human[int(x) - 1] for x in f.readlines()]

    examples = list(zip(range(len(labels)), image_files, labels))

    np.random.seed(0)
    np.random.shuffle(examples)

    if n_examples > 0:
        examples = examples[:n_examples]
    elif n_batches > 0:
        examples = examples[:batch_size * n_batches]
    else:
        print('using all images')

    def batch_loader(batch):
        batch = list(map(list, zip(*batch)))
        ids, xs, ys = batch
        return (ids, [viz.pil_loader(x) for x in xs], ys)

    batch_indices = list(range(0, len(examples), batch_size))
    batches = [examples[i: i + batch_size] for i in batch_indices]
    batches = map(batch_loader, batches)
    print('image loaded', len(batch_indices))
    return model, batches


def run_attack(explainers, attackers):
    model, batches = setup_imagenet(n_batches=1)
    results = []
    for batch_idx, batch in enumerate(batches):
        ids, xs, ys = batch
        xs = torch.stack([transf(x) for x in xs]).cuda()
        rows = attack_batch(model, xs, explainers, attackers)
        for i, row in enumerate(rows):
            rows[i]['idx'] = ids[row[i]['idx']]
        results += rows
    df = pd.DataFrame(results)
    df.drop(['idx'], axis=1)
    print(df.groupby(['attacker', 'explainer']).mean())


def plot_matrix(matrix, filename):
    '''Each entry in the matrix should be a dictionary of:
    image: an image ready to be plotted by imshow
    cmap: color map or None
    title: title
    rotate_title: if True show on the left of the image
    '''
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    f, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    for i, row in enumerate(matrix):
        for j, c in enumerate(row):
            image = c.get('image', None)
            cmap = c.get('cmap', None)
            title = c.get('title', None)
            rotate_title = c.get('rotate_title', False)
            if image is None:
                ax[i, j].imshow(np.zeros((244, 244)), cmap='gray')
            if cmap is None:
                ax[i, j].imshow(image)
            else:
                ax[i, j].imshow(image, cmap=cmap)
            if title is not None:
                title_fontsize = 20
                if rotate_title:
                    ax[i, j].set_title(title, rotation='vertical',
                                       x=-0.1, y=0.5, fontsize=title_fontsize)
                else:
                    ax[i, j].set_title(c['title'], fontsize=title_fontsize)
            ax[i, j].set_axis_off()
    f.tight_layout()
    f.savefig(filename)


def get_saliency_maps(model, batches, explainers, attackers):
    '''Collect saliency mappings.
    Returns a dictionary keyed by the example ids, each entry has:
    idx: example id (real id)
    attacker: name of the attacker
    explainer: name of the explainer
    perturbed: perturbed input
    saliency_1: saliency without perturbation
    saliency_2: saliency with perturbation
    '''
    results = defaultdict(dict)
    all_images = dict()
    all_labels = dict()
    all_ids = []
    for batch_idx, batch in enumerate(batches):
        ids, images, labels = batch
        for idx, img, lab in zip(ids, images, labels):
            all_images[idx] = img
            all_labels[idx] = lab
            all_ids.append(idx)
        xs = torch.stack([transf(x) for x in images]).cuda()
        rows = attack_batch(model, xs, explainers, attackers,
                            return_saliency=True)
        for i, row in enumerate(rows):
            idx = ids[row['idx']]
            attacker = row['attacker']
            explainer = row['explainer']
            row['saliency_1'] = np.abs(row['saliency_1']).sum(0)
            row['saliency_2'] = np.abs(row['saliency_2']).sum(0)
            ptb = np.array(row['perturbed']).swapaxes(0, 2).swapaxes(0, 1)
            row['perturbed'] = np.uint8(ptb * 255)
            row['idx'] = idx
            results[idx][(attacker, explainer)] = row
    return results, all_ids, all_images, all_labels


def plot_explainer_attacker():
    n_examples = 4

    attackers = [
        ('Original', EmptyAttack()),  # empty attacker so perturbed = original
        ('Ghorbani', GhorbaniAttack()),
        ('Random', ScaledNoiseAttack()),
    ]

    explainers = [
        ('Sparse (l1=0.5)', SparseExplainer(lambda_l1=0.5)),
        ('Robust (l1=0.5)', RobustSparseExplainer(lambda_l1=0.5)),
        ('Vanilla', VanillaGradExplainer()),
        ('SmoothGrad', SmoothGradExplainer()),
        ('IntegratedGrad', IntegrateGradExplainer()),
    ]

    model, batches = setup_imagenet(n_examples=n_examples)
    results, ids, images, labels = get_saliency_maps(model, batches,
                                                     explainers, attackers)
    assert len(results) == n_examples

    # construct the matrix to be plotted
    matrix = []
    ids = sorted(ids)  # ordered by real ids
    for idx in ids:
        label = '"{}"'.format(labels[idx])
        for i, (attacker, _) in enumerate(attackers):
            cell = results[idx][(attacker, explainers[0][0])]
            # show label on top if empty attacker
            # otherwise show attack name on the left
            row = [{
                    'image': cell['perturbed'],
                    'title': label if i == 0 else attacker,
                    'rotate_title': False if i == 0 else True,
            }]
            for explainer, _ in explainers:
                # only show explainer label on top of the row of original
                # example without perturbation
                cell = results[idx][(attacker, explainer)]
                s1 = cell['saliency_1'] 
                s2 = cell['saliency_2'] 
                row.append({
                    'image': s1 if i == 0 else s2,
                    'cmap': 'gray',
                    'title': explainer if i == 0 else None,
                })
            matrix.append(row)

    plot_matrix(matrix, 'figures/explainer_attacker.pdf')


def plot_l1_l2():
    n_examples = 4

    attackers = [
        # ('Original', EmptyAttack()),  # empty attacker so perturbed = original
        ('Ghorbani', GhorbaniAttack()),
    ]

    l1s = [0, 0.1, 0.5, 1, 10, 100]
    l2s = [10, 1e2, 1e3, 1e4, 1e5]

    explainers = []
    for l1 in l1s:
        for l2 in l2s:
            explainers.append(
                (
                    (l1, l2),  # use the combination as name
                    SparseExplainer(lambda_l1=l1, lambda_l2=l2)
                )
            )
            
    model, batches = setup_imagenet(n_examples=n_examples)
    results, ids, images, labels = get_saliency_maps(model, batches,
                                                     explainers, attackers)
    assert len(results) == n_examples
    resize = transforms.Resize((224, 224))
    # construct the matrix to be plotted
    matrix = []
    ids = sorted(ids)  # ordered by real ids
    for idx in ids:
        # label = '"{}"'.format(labels[idx])
        attacker = attackers[0][0]
        image = resize(images[idx])
        for i, l2 in enumerate(l2s):
            row = [{
                'image': image,
                'title': 'l2={}'.format(l2),
                'rotate_title': True
            }]
            for l1 in l1s:
                # only show explainer label on top of the row of original
                # example without perturbation
                cell = results[idx][(attacker, (l1, l2))]
                row.append({
                    'image': cell['saliency_1'],
                    'cmap': 'gray',
                    'title': 'l1={}'.format(l1) if i == 0 else None,
                })
            matrix.append(row)
    plot_matrix(matrix, 'figures/l1_l2.pdf')


if __name__ == '__main__':
    attackers = [
        ('Ghorbani', GhorbaniAttack()),
        ('Random', ScaledNoiseAttack()),
    ]

    explainers = [
        ('Sparse (l1=0.5)', SparseExplainer(lambda_l1=0.5)),
        ('Robust (l1=0.5)', RobustSparseExplainer(lambda_l1=0.5)),
        ('Vanilla', VanillaGradExplainer()),
        ('SmoothGrad', SmoothGradExplainer()),
        ('IntegratedGrad', IntegrateGradExplainer()),
    ]

    # run_attack(explainers, attackers)
    # plot_explainer_attacker()
    plot_l1_l2()