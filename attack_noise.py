import numpy as np
import pytablewriter
from scipy.stats import entropy, spearmanr
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import viz
import utils
from create_explainer import get_explainer
from preprocess import get_preprocess
from param_matrix import get_saliency

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class NoiseAttack(object):
    def __init__(self, epsilon=2.0/255.0):
        self.epsilon = epsilon

    def get_noisy(self, inp):
        grad_sign = 2 * np.random.randint(2, size=(3, 224, 224)) - 1   # random -1,1 matrix
        perturbed_inp = inp.data.squeeze().cpu().numpy() + grad_sign * self.epsilon
        perturbed_inp = np.clip(perturbed_inp, 0, 1)
        return Variable(torch.from_numpy(perturbed_inp).float().cuda(),
                        requires_grad=True)


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


def run_noise():
    model_methods = [
        ['resnet50', 'vanilla_grad', 'camshow', None],
        ['resnet50', 'grad_x_input', 'camshow', None],
        ['resnet50', 'smooth_grad', 'camshow', None],
        ['resnet50', 'integrate_grad', 'camshow', None],
        # ['resnet50', 'deconv', 'imshow', None],
        # ['resnet50', 'guided_backprop', 'imshow', None],
        # ['resnet50', 'gradcam', 'camshow', None],
        ['resnet50', 'sparse', 'camshow', None],
        ['resnet50', 'sparse_integrate_grad', 'camshow', None],
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

    noise_attacker = NoiseAttack()

    # attacker = HessianAttack(model, hessian_coefficient=1,
    #                          lambda_l1=0, lambda_l2=1e5,
    #                          n_iterations=10)

    inp = utils.cuda_var(transf(raw_img).unsqueeze(0), requires_grad=True)
    noisy_inp = noise_attacker.get_noisy(inp)

    rows = []
    for model_name, method_name, viz_style, kwargs in model_methods:
        inp_org = transf(raw_img)

        explainer = get_explainer(model, method_name, kwargs)
        transf = get_preprocess(model_name, method_name)

        filename_o = '{}.{}.org.png'.format(output_path, method_name)
        filename_g = '{}.{}.gho.png'.format(output_path, method_name)
        filename_m = '{}.{}.mask.png'.format(output_path, method_name)

        saliency_org = get_saliency(model, explainer, deepcopy(inp_org),
                                    raw_img, model_name, method_name,
                                    viz_style, filename_o)
        # inp_gho, protected = fuse(inp_org, delta.clone(), saliency_org,
        #                           epsilon=5e-2, gamma=    0)
        # saliency_gho = get_saliency(model, explainer, inp_gho, raw_img,
        #                             model_name, method_name, viz_style,
        #                             filename_g)

        saliency_gho = get_saliency(model, explainer,
                                    deepcopy(noisy_inp.data),
                                    raw_img, model_name, method_name,
                                    viz_style, filename_g)

        # protected = utils.upsample(protected.unsqueeze(0),
        #                            (raw_img.height, raw_img.width))
        # protected = protected.cpu().numpy()
        # protected = np.abs(protected).max(axis=1).squeeze()
        # plt.imshow(protected, cmap='jet')
        # plt.axis('off')
        # plt.savefig(filename_m)

        print(method_name)
        print(get_prediction(model, inp_org).data.cpu().numpy()[0])
        # print(get_prediction(model, inp_gho).data.cpu().numpy()[0])
        print(get_prediction(model, noisy_inp.data).data.cpu().numpy()[0])
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
    run_noise()
