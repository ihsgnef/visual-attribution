import numpy as np
import pytablewriter
from scipy.stats import entropy, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F

import viz
import utils
from create_explainer import get_explainer
from preprocess import get_preprocess
from param_matrix import get_saliency

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class HessianAttack(object):

    def __init__(self, model, hessian_coefficient=1,
                 lambda_l1=1e4, lambda_l2=1e4,
                 n_iterations=10):
        self.model = model
        self.hessian_coefficient = hessian_coefficient
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.n_iterations = n_iterations

    def attack(self, inp, ind=None, return_loss=False):
        batch_size, n_chs, img_height, img_width = inp.shape
        img_size = img_height * img_width
        delta = torch.zeros((batch_size, n_chs, img_size)).cuda()
        delta = nn.Parameter(delta, requires_grad=True)
        optimizer = torch.optim.SGD([delta], lr=0.1)
        # optimizer = torch.optim.Adam([delta], lr=0.0001)
        for i in range(self.n_iterations):
            output = self.model(inp)
            ind = output.max(1)[1]
            out_loss = F.cross_entropy(output, ind)
            inp_grad, = torch.autograd.grad(out_loss, inp, create_graph=True)
            inp_grad = inp_grad.view((batch_size, n_chs, img_size))
            hessian_delta_vp, = torch.autograd.grad(
                    inp_grad.dot(delta).sum(), inp, create_graph=True)
            hessian_delta_vp = hessian_delta_vp.view(
                    (batch_size, n_chs, img_size))
            taylor_1 = inp_grad.dot(delta).sum()
            taylor_2 = 0.5 * delta.dot(hessian_delta_vp).sum()
            l1_term = F.l1_loss(delta, torch.zeros_like(delta))
            l2_term = F.mse_loss(delta, torch.zeros_like(delta))
            loss = taylor_1 - self.hessian_coefficient * taylor_2
            loss += self.lambda_l1 * l1_term + self.lambda_l2 * l2_term
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        delta = delta.view((batch_size, n_chs, img_height, img_width))
        return delta.data


def get_prediction(model, inp):
    inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)
    output = model(inp)
    ind = output.max(1)[1]
    return ind


def saliency_correlation(saliency_1, saliency_2):
    saliency_1 = saliency_1.cpu().numpy()
    saliency_2 = saliency_2.cpu().numpy()
    saliency_1 = np.abs(saliency_1).max(axis=1).squeeze()
    saliency_2 = np.abs(saliency_2).max(axis=1).squeeze()
    saliency_1 = saliency_1.ravel()
    saliency_2 = saliency_2.ravel()

    saliency_1 -= saliency_1.min()
    saliency_1 /= (saliency_1.max() + 1e-20)
    saliency_2 -= saliency_2.min()
    saliency_2 /= (saliency_2.max() + 1e-20)

    # return entropy(saliency_1, saliency_2)
    return spearmanr(saliency_1, saliency_2)


def fuse(inp, delta, mask, epsilon=1e-2, gamma=3e-1):
    '''use saliency as a mask and fuse inp with delta'''
    inp = inp.cpu().squeeze(0).numpy()
    delta = delta.cpu().squeeze(0).numpy()
    mask = mask.cpu().squeeze(0).numpy()
    n_chs, img_height, img_width = inp.shape
    img_size = img_height * img_width
    inp = inp.reshape(n_chs, img_size)
    delta = delta.reshape(n_chs, img_size)
    mask = mask.reshape(n_chs, img_size)

    mask = np.abs(mask).max(axis=0)
    mask_idx = mask.argsort()[::-1]  # descend
    protected_idx = mask_idx[:int(gamma * mask_idx.size)]
    print(protected_idx.size)
    delta[:, protected_idx] = 0
    fused = np.clip(inp + epsilon * delta, 0, 1)
    fused = fused.reshape(n_chs, img_height, img_width)
    fused = torch.FloatTensor(fused)

    protected = np.ones_like(inp)
    protected[:, protected_idx] = 0
    protected = protected.reshape(n_chs, img_height, img_width)
    protected = torch.FloatTensor(protected)
    return fused, protected


def run_hessian():
    model_methods = [
        ['resnet50', 'vanilla_grad', 'camshow', None],
        ['resnet50', 'grad_x_input', 'camshow', None],
        ['resnet50', 'vanilla_grad', 'camshow', None],
        ['resnet50', 'smooth_grad', 'camshow', None],
        ['resnet50', 'vanilla_grad', 'camshow', None],
        ['resnet50', 'integrate_grad', 'camshow', None],
        ['resnet50', 'vanilla_grad', 'camshow', None],
        # ['resnet50', 'deconv', 'imshow', None],
        # ['resnet50', 'guided_backprop', 'imshow', None],
        # ['resnet50', 'gradcam', 'camshow', None],
        ['resnet50', 'sparse', 'camshow', {'n_iterations': 50}],
        # ['resnet50', 'excitation_backprop', 'camshow', None],
        # ['resnet50', 'contrastive_excitation_backprop', 'camshow', None],
        # ['vgg16', 'pattern_net', 'imshow', None],
        # ['vgg16', 'pattern_lrp', 'camshow', None],
    ]

    input_path = 'examples/tricycle.png'
    output_path = 'output/tricycle'
    model_name = 'resnet50'
    method_name = 'sparse'
    viz_style = 'camshow'
    raw_img = viz.pil_loader(input_path)
    transf = get_preprocess(model_name, method_name)
    model = utils.load_model(model_name)
    model.cuda()

    attacker = HessianAttack(model, hessian_coefficient=1,
                             lambda_l1=0, lambda_l2=0,
                             n_iterations=10)

    inp = utils.cuda_var(transf(raw_img).unsqueeze(0), requires_grad=True)
    inp_org = transf(raw_img)

    delta = attacker.attack(inp)
    delta = (delta - delta.min()) / (delta.max() + 1e-20)

    rows = []
    for model_name, method_name, viz_style, kwargs in model_methods:
        explainer = get_explainer(model, method_name, kwargs)
        transf = get_preprocess(model_name, method_name)

        filename_o = '{}.{}.org.png'.format(output_path, method_name)
        filename_g = '{}.{}.gho.png'.format(output_path, method_name)
        filename_m = '{}.{}.mask.png'.format(output_path, method_name)

        saliency_org = get_saliency(model, explainer, inp_org, raw_img,
                                    model_name, method_name, viz_style,
                                    filename_o)

        inp_gho, protected = fuse(inp_org, delta.clone(), saliency_org,
                                  epsilon=5e-2, gamma=0.3)
        saliency_gho = get_saliency(model, explainer, inp_gho, raw_img,
                                    model_name, method_name, viz_style,
                                    filename_g)

        protected = utils.upsample(protected.unsqueeze(0),
                                   (raw_img.height, raw_img.width))
        protected = protected.cpu().numpy()
        protected = np.abs(protected).max(axis=1).squeeze()
        plt.imshow(protected, cmap='jet')
        plt.axis('off')
        plt.savefig(filename_m)

        print(method_name)
        print(get_prediction(model, inp_org).data.cpu().numpy()[0])
        print(get_prediction(model, inp_gho).data.cpu().numpy()[0])
        print(saliency_correlation(saliency_org, saliency_gho))
        print()

        files = [filename_o, filename_g, filename_m]
        rows.append([method_name] + ['![]({})'.format(x) for x in files])

    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = "ghorbani attack"
    writer.header_list = ['method', 'original saliency', 'perturbed saliency',
                          'protected']
    writer.value_matrix = rows
    with open('{}.ghorbani.md'.format(output_path), 'w') as f:
        writer.stream = f
        writer.write_table()


if __name__ == '__main__':
    run_hessian()
