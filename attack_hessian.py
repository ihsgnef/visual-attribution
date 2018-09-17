import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import viz
import utils
from create_explainer import get_explainer
from preprocess import get_preprocess


class NoiseAttack:

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def attack(self, inp):
        noise = 2 * np.random.randint(2, size=inp.shape) - 1
        noise = np.sign(noise) * self.epsilon
        perturbed = np.clip(inp.data.cpu().numpy() + noise, 0, 1)
        perturbed = torch.FloatTensor(perturbed)
        noise = torch.FloatTensor(noise)
        return perturbed, noise


class GhorbaniAttack:

    def __init__(self, model,
                 lambda_t1=1, lambda_t2=1,
                 lambda_l1=1e4, lambda_l2=1e4,
                 n_iterations=10, optim='sgd', lr=1e-2,
                 epsilon=2 / 255):
        self.model = model
        self.lambda_t1 = lambda_t1
        self.lambda_t2 = lambda_t2
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.n_iterations = n_iterations
        self.optim = optim.lower()
        self.lr = lr
        self.epsilon = epsilon

    def attack(self, inp):
        inp_org = inp.data.cpu().numpy()
        ind_org = self.model(inp).max(1)[1].data.cpu().numpy()

        batch_size, n_chs, img_height, img_width = inp.shape
        step_size = self.epsilon / self.n_iterations
        accu_perturb = np.zeros_like(inp_org)
        prev_perturb = np.zeros_like(inp_org)
        stopped = [False for _ in range(batch_size)]
        for i in range(self.n_iterations):
            inp = np.clip(inp_org + accu_perturb, 0, 1)
            inp = Variable(torch.from_numpy(inp).cuda(), requires_grad=True)
            output = self.model(inp)
            ind = output.max(1)[1]
            ind_new = ind.data.cpu().numpy()
            for batch_idx in range(batch_size):
                if ind_new[batch_idx] != ind_org[batch_idx]:
                    accu_perturb[batch_idx] = prev_perturb[batch_idx]
                    stopped[batch_idx] = True

            out_loss = F.cross_entropy(output, ind)
            inp_grad, = torch.autograd.grad(out_loss, inp, create_graph=True)
            inp_grad = inp_grad.view(batch_size, n_chs, -1)

            topk = inp_grad.abs().sort(dim=2, descending=True)[0]
            topk = topk[:, :, :1000].sum()
            delta, = torch.autograd.grad(-topk, inp)
            delta = delta.sign().data.cpu().numpy()

            for batch_idx in range(batch_size):
                if stopped[batch_idx]:
                    delta[batch_idx] = 0

            prev_perturb = accu_perturb.copy()
            accu_perturb = accu_perturb + step_size * delta

        perturbed = np.clip(inp_org + accu_perturb, 0, 1)
        perturbed = torch.FloatTensor(perturbed)
        accu_perturb = torch.FloatTensor(accu_perturb)
        # final check that all predictions remain
        inp = Variable(perturbed.cuda())
        ind_new = self.model(inp).max(1)[1].data.cpu().numpy()
        assert (ind_org == ind_new).all()
        return perturbed, accu_perturb


def saliency_correlation(s1, s2):
    # s1 and s2 are batched
    assert s1.shape == s2.shape
    assert s1.ndimension() == 3  # batch, height, width
    batch_size = s1.shape[0]
    s1 = s1.cpu().numpy().reshape(batch_size, -1)
    s2 = s2.cpu().numpy().reshape(batch_size, -1)
    scores = []
    for x1, x2 in zip(s1, s2):
        scores.append(spearmanr(x1, x2).correlation)
    return scores


def saliency_overlap(s1, s2):
    assert s1.shape == s2.shape
    batch_size = s1.shape[0]
    s1 = s1.cpu().numpy().reshape(batch_size, -1)
    s2 = s2.cpu().numpy().reshape(batch_size, -1)
    scores = []
    K = 1000
    for x1, x2 in zip(s1, s2):
        x1 = set(np.argsort(-x1)[:K])
        x2 = set(np.argsort(-x2)[:K])
        scores.append(len(x1.intersection(x2)) / K)
    return scores


transf = get_preprocess('resnet50', 'sparse')
model = utils.load_model('resnet50')
model.eval()
model.cuda()

configs = [
    ['sparse 1',
     {
        'lambda_t1': 1,
        'lambda_t2': 1,
        'lambda_l1': 0,
        'lambda_l2': 0,
        'n_iterations': 10,
        'optim': 'sgd',
        'lr': 0.1,
     }],
    ['sparse 2',
     {
        'lambda_t1': 1,
        'lambda_t2': 0,
        'lambda_l1': 0,
        'lambda_l2': 0,
        'n_iterations': 10,
        'optim': 'sgd',
        'lr': 0.1,
     }],
    ['vanilla_grad', None],
    ['grad_x_input', None],
    ['smooth_grad', None],
    ['integrate_grad', None],
]

attackers = [
    (GhorbaniAttack(
        model,
        lambda_t1=0,
        lambda_t2=1,
        lambda_l1=0, 
        lambda_l2=0,
        n_iterations=30, 
        optim='sgd', 
        lr=1e-2, 
        epsilon=2 / 255), 'gho'),
    (NoiseAttack(epsilon=2 / 255), 'rnd'),
]


def aggregate(saliency, image):
    '''combine saliency mapping with image'''
    saliency = saliency.cpu()
    image = image.cpu()
    return viz.VisualizeImageGrayscale(saliency)
    # return saliency.max(dim=1)[0]
    # return saliency.abs().max(dim=1)[0]
    # return saliency.sum(dim=1)
    # return saliency.abs().sum(dim=1)
    # return (saliency * image).sum(dim=1)
    # return (saliency * image).abs().sum(dim=1)
    # return (saliency * image).abs().max(dim=1)[0]
    # return (saliency * image).max(dim=1)[0]


def perturb(image, delta, mask=None, flip=False):
    '''perturb image with delta within mask
    create zero-one mask where locations with high mask value get one
    if flip is True, flip the sign of zero-one mask
    '''
    assert len(image.shape)
    assert mask.ndimension() == 3  # batch, height, width
    assert image.shape == delta.shape
    batch_size, n_chns, height, width = image.shape
    if isinstance(image, Variable):
        image = image.data
    image = image.cpu().numpy()
    delta = delta.cpu().numpy()
    mask = mask.cpu().numpy()
    mask = mask.reshape(batch_size, -1)
    zero_mask = np.zeros_like(mask)
    K = 1000
    mask = np.argsort(-mask, axis=1)[:, :K]
    for i in range(batch_size):
        zero_mask[i][mask[i]] = 1
    zero_mask = zero_mask.reshape(batch_size, height, width)
    zero_mask = np.expand_dims(zero_mask, 1)
    if flip:
        zero_mask = 1 - zero_mask
    perturbed = image + delta * zero_mask
    return torch.from_numpy(perturbed).cuda()


def run_hessian(raw_images):
    '''construct attacks'''
    attacks = []
    for atk, attack_name in attackers:
        images = torch.stack([transf(x) for x in raw_images])
        images = Variable(images.cuda(), requires_grad=True)
        perturbed, delta = atk.attack(images)
        attacks.append((perturbed, delta, attack_name))

    '''run saliency methods'''
    results = []
    for method_name, kwargs in configs:
        explainer = get_explainer(model, method_name, kwargs)
        images = torch.stack([transf(x) for x in raw_images])
        inputs = Variable(images.clone().cuda(), requires_grad=True)
        saliency_1 = aggregate(explainer.explain(inputs), inputs.data)

        for perturbed, delta, attack_name in attacks:
            # inputs = perturbed.clone()
            batch_size = images.shape[0]

            inputs = perturbed.clone()
            inputs = Variable(inputs.cuda(), requires_grad=True)
            saliency_2 = aggregate(explainer.explain(inputs), inputs.data)

            inputs = perturb(images, delta, saliency_1)
            inputs = Variable(inputs.cuda(), requires_grad=True)
            saliency_3 = aggregate(explainer.explain(inputs), inputs.data)

            inputs = perturb(images, delta, saliency_1, flip=True)
            inputs = Variable(inputs.cuda(), requires_grad=True)
            saliency_4 = aggregate(explainer.explain(inputs), inputs.data)

            scores = [
                saliency_correlation(saliency_1, saliency_2),
                saliency_correlation(saliency_1, saliency_3),
                saliency_correlation(saliency_1, saliency_4),
                saliency_overlap(saliency_1, saliency_2),
                saliency_overlap(saliency_1, saliency_3),
                saliency_overlap(saliency_1, saliency_4),
                saliency_overlap(saliency_1, aggregate(delta, inputs.data)),
                saliency_overlap(saliency_2, aggregate(delta, inputs.data)),
                saliency_overlap(saliency_3, aggregate(delta, inputs.data)),
                saliency_overlap(saliency_4, aggregate(delta, inputs.data)),
            ]
            scores = list(map(list, zip(*scores)))

            for i in range(batch_size):
                results.append([
                    method_name,
                    attack_name,
                ] + scores[i])
    return results


if __name__ == '__main__':
    image_path = '/fs/imageNet/imagenet/ILSVRC_val/**/*.JPEG'
    image_files = list(glob.iglob(image_path, recursive=True))
    batch_size = 16
    indices = list(range(0, len(image_files), batch_size))
    results = []
    for batch_idx, start in enumerate(indices):
        if batch_idx > 1:
            break
        batch = image_files[start: start + batch_size]
        raw_images = [viz.pil_loader(x) for x in batch]
        results += run_hessian(raw_images)
    n_scores = len(results[0]) - 2  # number of different scores
    print(n_scores)
    # results = list(map(list, zip(*results)))
    columns = (
        ['method', 'attack'] +
        ['score_{}'.format(i) for i in range(n_scores)]
    )
    results = pd.DataFrame(results, columns=columns)
    results = results.groupby(['method', 'attack']).mean()
    print(results)
