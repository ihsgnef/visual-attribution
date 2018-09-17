import itertools
import pytablewriter
import numpy as np

import viz
import utils
from create_explainer import get_explainer
from preprocess import get_preprocess
from explainer.sparse import SparseExplainer
from viz import VisualizeImageGrayscale

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def get_saliency(model, explainer, inp, raw_img,
                 model_name, method_name, viz_style, filename):

    inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)
    saliency = explainer.explain(inp, None)
    # smap = utils.upsample(saliency, (raw_img.height, raw_img.width))
    # smap = smap.cpu().numpy()

    # if viz_style == 'camshow':
    #     viz.plot_cam(np.abs(smap).max(axis=1).squeeze(),
    #                  raw_img, 'jet', alpha=0.5)
    # else:
    #     if model_name == 'googlenet' or method_name == 'pattern_net':
    #         smap = smap.squeeze()[::-1].transpose(1, 2, 0)
    #     else:
    #         smap = smap.squeeze().transpose(1, 2, 0)
    #     smap -= smap.min()
    #     smap /= (smap.max() + 1e-20)
    #     plt.imshow(smap, cmap='gray')
    # plt.axis('off')
    # plt.savefig(filename)

    saliency = viz.VisualizeImageGrayscale(saliency)
    viz.ShowGrayscaleImage(saliency.cpu().numpy()[0])
    plt.savefig(filename)
    plt.axis('off')
    return saliency

def lambda_l1_n_iter(input_path, output_path):
    model_name = 'resnet18' #######################################resnet50
    method_name = 'sparse'
    viz_style = 'imshow'
    raw_img = viz.pil_loader(input_path)
    transf = get_preprocess(model_name, method_name)
    model = utils.load_model(model_name)
    model.cuda()

    lambda_l1s = [0, 1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    n_iterations = [10]#list(range(1, 15))
    all_configs = list(itertools.product(lambda_l1s, n_iterations))
    all_configs = [{'lambda_l1': ll1, 'n_iterations': n_iter}
                   for ll1, n_iter in all_configs]

    for cfg_id, cfg in enumerate(all_configs):
        explainer = SparseExplainer(model, **cfg)
        filename = '{}.l1_niter.{}.png'.format(output_path, cfg_id)
        all_configs[cfg_id]['output_path'] = filename
        inp = transf(raw_img)
        get_saliency(model, explainer, inp, raw_img, model_name,
                     method_name, viz_style, filename)

    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = "lambda_l1 vs n_iterations"
    writer.header_list = [''] + ['n_iter: {}'.format(x) for x in n_iterations]
    writer.value_matrix = []
    for i, lambda_l1 in enumerate(lambda_l1s):
        row = ['lambda l1: {}'.format(lambda_l1)]
        for j, n_iter in enumerate(n_iterations):
            cfg = all_configs[i * len(n_iterations) + j]
            row.append('![]({})'.format(cfg['output_path']))
        writer.value_matrix.append(row)
    with open('{}.l1_niter.md'.format(output_path), 'w') as f:
        writer.stream = f
        writer.write_table()


def lambda_l1_l2(input_path, output_path):
    model_name = 'resnet50'
    method_name = 'sparse'
    viz_style = 'imshow'
    raw_img = viz.pil_loader(input_path)
    transf = get_preprocess(model_name, method_name)
    model = utils.load_model(model_name)
    model.cuda()

    lambda_l1s = [0, 1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    lambda_l2s = [0, 1, 1e2, 1e3, 1e4, 1e5, 1e6]
    all_configs = list(itertools.product(lambda_l1s, lambda_l2s))
    all_configs = [{'lambda_l1': ll1, 'lambda_l2': ll2}
                   for ll1, ll2 in all_configs]

    for cfg_id, cfg in enumerate(all_configs):
        explainer = SparseExplainer(model, **cfg)
        filename = '{}.l1_l2.{}.png'.format(output_path, cfg_id)
        all_configs[cfg_id]['output_path'] = filename
        inp = transf(raw_img)
        get_saliency(model, explainer, inp, raw_img, model_name,
                     method_name, viz_style, filename)

    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = "lambda_l1 vs lambda l2"
    writer.header_list = [''] + ['lambda_l2: {}'.format(x) for x in lambda_l2s]
    writer.value_matrix = []
    for i, lambda_l1 in enumerate(lambda_l1s):
        row = ['lambda l1: {}'.format(lambda_l1)]
        for j, lambda_l2 in enumerate(lambda_l2s):
            cfg = all_configs[i * len(lambda_l2s) + j]
            assert cfg['lambda_l1'] == lambda_l1
            assert cfg['lambda_l2'] == lambda_l2
            # row.append('<img src="{}">'.format(cfg['output_path']))
            row.append('![]({})'.format(cfg['output_path']))
        writer.value_matrix.append(row)
    with open('{}.l1_l2.md'.format(output_path), 'w') as f:
        writer.stream = f
        writer.write_table()


def baselines(input_path, output_path):
    model_methods = [
        ['resnet50', 'vanilla_grad', 'imshow', None],
        ['resnet50', 'grad_x_input', 'imshow', None],
        ['resnet50', 'saliency', 'imshow', None],
        ['resnet50', 'smooth_grad', 'imshow', None],
        # ['resnet50', 'deconv', 'imshow', None],
        # ['resnet50', 'guided_backprop', 'imshow', None],
        ['resnet50', 'gradcam', 'camshow', None],
        # ['resnet50', 'excitation_backprop', 'camshow', None],
        # ['resnet50', 'contrastive_excitation_backprop', 'camshow', None],
        # ['vgg16', 'pattern_net', 'imshow', None],
        # ['vgg16', 'pattern_lrp', 'camshow', None],
    ]

    table_row = ['![]({})'.format(input_path)]
    raw_img = viz.pil_loader(input_path)
    for model_name, method_name, viz_style, kwargs in model_methods:
        transf = get_preprocess(model_name, method_name)
        model = utils.load_model(model_name)
        model.cuda()
        explainer = get_explainer(model, method_name, kwargs)
        filename = '{}.{}.png'.format(output_path, method_name)
        inp = transf(raw_img)
        get_saliency(model, explainer, inp, raw_img, model_name,
                     method_name, viz_style, filename)
        table_row.append('![]({})'.format(filename))

    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = "baselines"
    writer.header_list = ['input'] + [row[1] for row in model_methods]
    writer.value_matrix = [table_row]
    with open('{}.baselines.md'.format(output_path), 'w') as f:
        writer.stream = f
        writer.write_table()


if __name__ == '__main__':
    # baselines(input_path='examples/tricycle.png',
    #           output_path='output/tricycle')
    #lambda_l1_n_iter(input_path='examples/fox.png',
    #                 output_path='output/fox')
    lambda_l1_l2(input_path='examples/fox.png',
                 output_path='output/fox')
