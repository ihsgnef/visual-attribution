import torch
import pytablewriter

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import viz
import utils
from create_explainer import get_explainer
from preprocess import get_preprocess
from explainer.sparse import SparseExplainer
from explainer.backprop import VanillaGradExplainer
from param_matrix import get_saliency


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


def main():
    model_methods = [
        ['resnet50', 'vanilla_grad', 'camshow', None],
        ['resnet50', 'grad_x_input', 'camshow', None],
        ['resnet50', 'saliency', 'camshow', None],
        ['resnet50', 'smooth_grad', 'camshow', None],
        # ['resnet50', 'deconv', 'imshow', None],
        # ['resnet50', 'guided_backprop', 'imshow', None],
        ['resnet50', 'gradcam', 'camshow', None],
        ['resnet50', 'sparse', 'camshow', None],
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
                             lambda_l1=0, lambda_l2=1e5,
                             n_iterations=10)
    inp = transf(raw_img)
    inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)
    delta = attacker.attack(inp)
    delta = delta.cpu()
    delta = (delta - delta.min()) / (delta.max() + 1e-20)

    inp_org = transf(raw_img)
    inp_gho = 0.9 * inp_org + 0.1 * delta.cpu().squeeze(0)
    img_org = transforms.ToPILImage()(inp_org)
    img_gho = transforms.ToPILImage()(inp_gho)
    filename_o = '{}.inp_org.png'.format(output_path)
    filename_g = '{}.inp_gho.png'.format(output_path)
    img_org.resize((raw_img.height, raw_img.width)).save(filename_o)
    img_gho.resize((raw_img.height, raw_img.width)).save(filename_g)
    print(get_prediction(model, transf(img_org)))
    print(get_prediction(model, transf(img_gho)))

    first_row = ['![]({})'.format(filename_o)]
    second_row = ['![]({})'.format(filename_g)]
    for model_name, method_name, viz_style, kwargs in model_methods:
        transf = get_preprocess(model_name, method_name)
        model = utils.load_model(model_name)
        model.cuda()
        explainer = get_explainer(model, method_name, kwargs)
        input_original = transf(img_org)
        input_perturbd = transf(img_gho)
        filename_o = '{}.{}.org.png'.format(output_path, method_name)
        filename_g = '{}.{}.gho.png'.format(output_path, method_name)
        get_saliency(model, explainer, input_original, raw_img,
                     model_name, method_name, viz_style, filename_o)
        get_saliency(model, explainer, input_perturbd, raw_img,
                     model_name, method_name, viz_style, filename_g)
        first_row.append('![]({})'.format(filename_o))
        second_row.append('![]({})'.format(filename_g))

    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = "ghorbani attack"
    writer.header_list = ['input'] + [row[1] for row in model_methods]
    writer.value_matrix = [first_row, second_row]
    with open('{}.ghorbani.md'.format(output_path), 'w') as f:
        writer.stream = f
        writer.write_table()


if __name__ == '__main__':
    main()
