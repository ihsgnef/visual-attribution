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
    for x1, x2 in zip(s1, s2):
        x1 = np.argsort(x1)[::-1]
        x2 = np.argsort(x2)[::-1]
        x1 = set(x1[:1000])
        x2 = set(x2[:1000])
        scores.append(len(x1.intersection(x2)) / 1000)
    return scores


def run_hessian(raw_images):
    transf = get_preprocess('resnet50', 'sparse')
    model = utils.load_model('resnet50')
    model.eval()
    model.cuda()

    sparse_args = {
        'lambda_t1': 1,
        'lambda_t2': 1,
        'lambda_l1': 0,
        'lambda_l2': 1e4,
        'n_iterations': 10,
        'optim': 'sgd',
        'lr': 1e-1,
    }

    configs = [
        ['sparse', sparse_args],
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
            epsilon=2 / 255
        ), 'gho'),
        (NoiseAttack(epsilon=2 / 255), 'rnd'),
    ]

    '''construct attacks'''
    attacks = []
    for atk, attack_name in attackers:
        inputs = torch.stack([transf(x) for x in raw_images])
        inputs = Variable(inputs.cuda(), requires_grad=True)
        perturbed, delta = atk.attack(inputs)
        attacks.append((perturbed, delta, attack_name))

    def aggregate(saliency, image):
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

    '''run saliency methods'''
    scores = dict()
    for method_name, kwargs in configs:
        explainer = get_explainer(model, method_name, kwargs)
        inputs = torch.stack([transf(x) for x in raw_images])
        inputs = Variable(inputs.cuda(), requires_grad=True)
        saliency_1 = explainer.explain(inputs, None)
        saliency_1 = aggregate(saliency_1, inputs.data)

        scores[method_name] = dict()
        for perturbed, delta, attack_name in attacks:
            inputs = Variable(perturbed.clone().cuda(), requires_grad=True)
            saliency_2 = explainer.explain(inputs, None)
            saliency_2 = aggregate(saliency_2, inputs.data)

            corr = saliency_correlation(saliency_1, saliency_2)
            over = saliency_overlap(saliency_1, saliency_2)

            delta = aggregate(delta, inputs.data)
            over_1 = saliency_overlap(saliency_1, delta)
            over_2 = saliency_overlap(saliency_2, delta)

            values = list(zip(corr, over, over_1, over_2))
            scores[method_name][attack_name] = values
    return scores


if __name__ == '__main__':
    image_path = '/fs/imageNet/imagenet/ILSVRC_val/**/*.JPEG'
    image_files = list(glob.iglob(image_path, recursive=True))[:64]
    batch_size = 16
    indices = list(range(0, len(image_files), batch_size))
    all_scores = None
    for batch_idx, start in enumerate(indices):
        if batch_idx > 3:
            break
        batch = image_files[start: start + batch_size]
        raw_images = [viz.pil_loader(x) for x in batch]
        scores = run_hessian(raw_images)
        if all_scores is None:
            all_scores = scores
            continue
        for method_name in scores:
            for attack_name, values in scores[method_name].items():
                all_scores[method_name][attack_name] += values

    results = {'method': [], 'attack': [], 'correlation': [], 'overlap': []}
    for method_name in all_scores:
        for attack_name, values in all_scores[method_name].items():
            values = list(map(list, zip(*values)))
            values = [np.mean(x) for x in values]
            results['method'].append(method_name)
            results['attack'].append(attack_name)
            results['correlation'].append(values[0])
            results['overlap'].append(values[1])
    results = pd.DataFrame(results)
    print(results)
