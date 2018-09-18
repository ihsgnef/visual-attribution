import json
import glob
import itertools
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


EPSILON = 2 / 255

configs = [
    ['sparse 1',
     {
        'lambda_t1': 1,
        'lambda_t2': 1,
        'lambda_l1': 100,
        'lambda_l2': 1e4,
        'n_iterations': 10,
        'optim': 'sgd',
        'lr': 0.1,
     }],
    ['sparse 2',
     {
        'lambda_t1': 0,
        'lambda_t2': 1,
        'lambda_l1': 100,
        'lambda_l2': 1e4,
        'n_iterations': 10,
        'optim': 'sgd',
        'lr': 0.1,
     }],
    ['sparse 3',
     {
        'lambda_t1': 1,
        'lambda_t2': 1,
        'lambda_l1': 0,
        'lambda_l2': 1e4,
        'n_iterations': 10,
        'optim': 'sgd',
        'lr': 0.1,
     }],
    ['vat 1', {'n_iterations': 1}],
    ['vat 10', {'n_iterations': 10}],
    ['vat 20', {'n_iterations': 20}],
    # ['vanilla_grad', None],
    # ['grad_x_input', None],
    # ['smooth_grad', None],
    # ['integrate_grad', None],
]

transf = get_preprocess('resnet50', 'sparse')


def zero_grad(x):
    if isinstance(x, Variable):
        if x.grad is not None:
            x.grad.data.zero_()
    elif isinstance(x, torch.nn.Module):
        for p in x.parameters():
                if p.grad is not None:
                    p.grad.data.zero_()
    

class NoiseAttack:

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def attack(self, inp):
        noise = 2 * np.random.randint(2, size=inp.shape) - 1
        noise = np.sign(noise) * self.epsilon
        perturbed = np.clip(inp.cpu().numpy() + noise, 0, 1)
        perturbed = torch.FloatTensor(perturbed)
        noise = torch.FloatTensor(noise)
        return perturbed, noise


class FGSM:

    def __init__(self, model, epsilon=2 / 255, n_iterations=10):
        self.model = model
        self.epsilon = epsilon
        self.n_iterations = n_iterations

    def attack(self, inp):
        inp_org = inp.clone()
        batch_size, n_chs, img_height, img_width = inp.shape
        step_size = self.epsilon / self.n_iterations
        for i in range(self.n_iterations):
            zero_grad(model)
            new_inp = Variable(inp, requires_grad=True)
            output = self.model(new_inp)
            out_loss = F.cross_entropy(output, output.max(1)[1])
            inp_grad, = torch.autograd.grad(
                out_loss, new_inp, create_graph=True)
            inp_grad = inp_grad.view(batch_size, n_chs, -1)
            delta = inp_grad.sign().data
            inp = torch.clamp(inp + step_size * delta, 0, 1)
        return inp, inp - inp_org


class ScaledNoiseAttack:

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def attack(self, inp):
        inp = inp.cpu().numpy()
        noise = 2 * np.random.randint(2, size=inp.shape) - 1
        noise = np.sign(noise) * self.epsilon
        perturbed = np.clip(inp + noise * inp, 0, 1)
        perturbed = torch.FloatTensor(perturbed)
        noise = torch.FloatTensor(noise)
        return perturbed, noise


class GhorbaniAttackFast:

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
        inp_org = inp.clone()
        batch_size, n_chs, img_height, img_width = inp.shape

        prev = inp.clone()
        delta = torch.zeros_like(inp)
        ind_org = self.model(Variable(inp)).max(1)[1].data

        step_size = self.epsilon / self.n_iterations
        stopped = [False for _ in range(batch_size)]
        for i in range(self.n_iterations):
            zero_grad(model)
            new_inp = Variable(inp, requires_grad=True)
            output = self.model(new_inp)
            ind = output.max(1)[1]
            out_loss = F.cross_entropy(output, ind)
            inp_grad, = torch.autograd.grad(
                out_loss, new_inp, create_graph=True)
            inp_grad = inp_grad.view(batch_size, n_chs, -1)

            topk = inp_grad.abs().sort(dim=2, descending=True)[0]
            topk = topk[:, :, :1000].sum()
            delta, = torch.autograd.grad(-topk, new_inp)
            delta = delta.sign().data

            # verify same prediction
            for bidx in range(batch_size):
                if stopped[bidx] or ind.data[bidx] != ind_org[bidx]:
                    inp[bidx] = prev[bidx]
                    delta[bidx].zero_()
                    stopped[bidx] = True

            if all(stopped):
                break

            prev = inp.clone()
            inp = torch.clamp(inp + step_size * delta, 0, 1)

        # final check that all predictions remain
        ind_new = self.model(Variable(inp)).max(1)[1].data
        assert (ind_org == ind_new).all()
        return inp, inp - inp_org


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
        inp_org = inp.cpu().numpy()
        inp = Variable(inp.cuda(), requires_grad=True)
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


def saliency_correlation(s1, s2, image):
    # s1 and s2 are batched
    s1 = aggregate(s1, image)
    s2 = aggregate(s2, image)
    assert s1.shape == s2.shape
    assert s1.ndimension() == 3  # batch, height, width
    batch_size = s1.shape[0]
    s1 = s1.cpu().numpy().reshape(batch_size, -1)
    s2 = s2.cpu().numpy().reshape(batch_size, -1)
    scores = []
    for x1, x2 in zip(s1, s2):
        scores.append(spearmanr(x1, x2).correlation)
    return scores


def channel_correlation(s1, s2, image):
    assert s1.shape == s2.shape
    assert s1.ndimension() == 4  # batch, 3, height, width
    batch_size = s1.shape[0]
    s1 = s1.cpu().numpy().reshape(batch_size, 3, -1)
    s2 = s2.cpu().numpy().reshape(batch_size, 3, -1)
    scores = []
    for x1, x2 in zip(s1, s2):
        scores.append(spearmanr(x1[k], x2[k]).correlation for k in range(3))
    scores = list(map(list, zip(*scores)))
    return scores


def saliency_overlap(s1, s2, image):
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


def binit(saliency):
    '''normalize saliency values then bin'''
    batch_size, _, height, width = saliency.shape
    s = saliency.cpu().numpy()
    s = np.abs(s).sum(1)
    s = s.reshape(batch_size, -1)
    # vmax = np.expand_dims(np.percentile(s, 90), 1)
    # vmin = np.expand_dims(np.percentile(s, 10), 1)
    vmax = np.expand_dims(np.max(s, 1), 1)
    vmin = np.expand_dims(np.min(s, 1), 1)
    s = (s - vmin) / (vmax - vmin)
    s = np.digitize(s, np.arange(0, 1, 1 / 10)) / 10
    s = np.clip(s, 0, 1)
    s = s.reshape(batch_size, height, width)
    return torch.FloatTensor(s)


def aggregate(saliency, image):
    '''combine saliency mapping with image
    from 4D (bsz, 3, h, w) to 3D (bsz, h, w)
    '''
    saliency = saliency.cpu()
    image = image.cpu()
    # return binit(saliency)
    # return viz.VisualizeImageGrayscale(saliency)
    # return saliency.abs().sum(dim=1)
    # # return saliency.max(dim=1)[0]
    # # return saliency.abs().max(dim=1)[0]
    # # return saliency.sum(dim=1)
    # # return (saliency * image).sum(dim=1)
    return (saliency * image).abs().sum(dim=1)
    # # return (saliency * image).abs().max(dim=1)[0]
    # # return (saliency * image).max(dim=1)[0]


def perturb(image, delta, mask, flip=False):
    '''perturb image with delta within mask
    create zero-one mask where locations with high mask value get one
    if flip is True, flip the sign of zero-one mask
    '''
    K = 10000
    assert len(image.shape)
    assert mask.ndimension() == 3  # batch, height, width
    assert image.shape == delta.shape
    batch_size, n_chs, height, width = image.shape
    image = image.cpu().numpy()
    delta = delta.cpu().numpy()
    mask = mask.cpu().numpy()
    mask = mask.reshape(batch_size, -1)
    zero_mask = np.zeros_like(mask)
    mask = np.argsort(-mask, axis=1)[:, :K]
    for i in range(batch_size):
        zero_mask[i][mask[i]] = 1
    zero_mask = zero_mask.reshape(batch_size, height, width)
    zero_mask = np.expand_dims(zero_mask, 1)
    if flip:
        zero_mask = 1 - zero_mask
    perturbed = image + delta * zero_mask
    return torch.from_numpy(perturbed).cuda()


def saliency_histogram(model, raw_images):
    results = []
    for method_name, kwargs in configs:
        explainer = get_explainer(model, method_name, kwargs)
        batch_size = len(raw_images)
        images = torch.stack([transf(x) for x in raw_images]).cuda()
        batch_size, n_chs, height, width = images.shape
        inputs = Variable(images.clone(), requires_grad=True)
        saliency = explainer.explain(inputs).cpu().numpy()
        saliency = saliency.reshape(batch_size, n_chs, height * width)
        # don't aggregate, look at channels separately and in combination
        for i in range(batch_size):
            for j, chn in enumerate(['R', 'G', 'B']):
                results.append([method_name, chn, saliency[i][j].tolist()])
            results.append([method_name, 'sum', saliency[i].sum(1).tolist()])
    return results


def attack_test(model, raw_images):
    attackers = [
        (GhorbaniAttackFast(
            model,
            lambda_t1=0,
            lambda_t2=1,
            lambda_l1=0,
            lambda_l2=0,
            n_iterations=30,
            optim='sgd',
            lr=1e-2,
            epsilon=EPSILON), 'gho'),
        # (FGSM(model, epsilon=EPSILON, n_iterations=10), 'fgsm'),
        # (NoiseAttack(epsilon=EPSILON), 'srnd'),
        (ScaledNoiseAttack(epsilon=EPSILON), 'rnd'),
    ]

    '''construct attacks'''
    attacks = []
    images = torch.stack([transf(x) for x in raw_images]).cuda()
    for atk, attack_name in attackers:
        perturbed, delta = atk.attack(images.clone())
        attacks.append((perturbed, delta, attack_name))

    '''run saliency methods'''
    results = []
    batch_size = images.shape[0]
    for method_name, kwargs in configs:
        explainer = get_explainer(model, method_name, kwargs)
        inputs = Variable(images.clone().cuda(), requires_grad=True)
        saliency_1 = explainer.explain(inputs)
        for perturbed, delta, attack_name in attacks:
            # unrestricted perturbation
            inputs = perturbed.clone()
            inputs = Variable(inputs.cuda(), requires_grad=True)
            saliency_2 = explainer.explain(inputs)

            # # only perturb highlighted region
            # inputs = perturb(images, delta, saliency_1)
            # inputs = Variable(inputs.cuda(), requires_grad=True)
            # saliency_3 = explainer.explain(inputs)

            # # perturb outside highlighted region
            # inputs = perturb(images, delta, saliency_1, flip=True)
            # inputs = Variable(inputs.cuda(), requires_grad=True)
            # saliency_4 = explainer.explain(inputs)

            scores = [
                saliency_correlation(saliency_1, saliency_2, inputs.data),
                *channel_correlation(saliency_1, saliency_2, inputs.data),
                # saliency_correlation(saliency_1, saliency_3, inputs.data),
                # saliency_correlation(saliency_1, saliency_4, inputs.data),

                saliency_overlap(saliency_1, saliency_2, inputs.data),
                # saliency_overlap(saliency_1, saliency_3, inputs.data),
                # saliency_overlap(saliency_1, saliency_4, inputs.data),

                # saliency_overlap(saliency_1, delta, inputs.data),
                # saliency_overlap(saliency_2, delta, inputs.data),
                # saliency_overlap(saliency_3, delta, inputs.data),
                # saliency_overlap(saliency_4, delta, inputs.data),
            ]
            scores = list(map(list, zip(*scores)))

            for i in range(batch_size):
                results.append([
                    method_name,
                    attack_name,
                ] + scores[i])
    return results


image_path = '/fs/imageNet/imagenet/ILSVRC_val/**/*.JPEG'
image_files = list(glob.iglob(image_path, recursive=True))
np.random.seed(0)
np.random.shuffle(image_files)
batch_size = 8
batch_indices = list(range(0, len(image_files), batch_size))
print('image path loaded')

model = utils.load_model('resnet50')
model.eval()
model.cuda()
print('model loaded')


def run_attack_short():
    results = []
    for batch_idx, start in enumerate(tqdm(batch_indices[:3])):
        batch = image_files[start: start + batch_size]
        raw_images = [viz.pil_loader(x) for x in batch]
        results += attack_test(model, raw_images)
    n_scores = len(results[0]) - 2  # number of different scores
    columns = (
        ['method', 'attack'] +
        ['score_{}'.format(i) for i in range(n_scores)]
    )
    df = pd.DataFrame(results, columns=columns)
    with open('output/results.812.json') as f:
        df = df.append(pd.DataFrame(json.load(f)), ignore_index=True)
    print(df.groupby(['attack', 'method']).mean())


def run_attack_long():
    # indices = batch_indices[:100]
    results = []
    indices = batch_indices
    n_scores = 4  # len(results[0]) - 2  # number of different scores
    columns = (
        ['method', 'attack'] +
        ['score_{}'.format(i) for i in range(n_scores)]
    )

    def check():
        df = pd.DataFrame(results, columns=columns)
        with open('output/results.{}.json'.format(batch_idx), 'w') as f:
            f.write(df.to_json())
        df = df.groupby(['attack', 'method']).mean()
        # print(df)

    for batch_idx, start in enumerate(tqdm(indices)):
        if batch_idx <= 812:
            continue
        if batch_idx % 20 == 0 and batch_idx > 0:
            check()
            results = []
        batch = image_files[start: start + batch_size]
        raw_images = [viz.pil_loader(x) for x in batch]
        results += attack_test(model, raw_images)
    if len(results) > 0:
        check()


def run_histogram():
    results = []
    for batch_idx, start in enumerate(batch_indices):
        if batch_idx > 0:
            break
        batch = image_files[start: start + batch_size]
        raw_images = [viz.pil_loader(x) for x in batch]
        results += saliency_histogram(model, raw_images)
    columns = (
        ['method', 'channel', 'saliency']
    )
    results = pd.DataFrame(results, columns=columns)
    results = results.groupby(['channel', 'method'])
    results = results.agg(lambda x: len(list(itertools.chain(*x))))
    print(results)
    with open('histogram.json', 'w') as f:
        json.dump(results.to_json(), f)


if __name__ == '__main__':
    run_attack_short()
    # run_histogram()
