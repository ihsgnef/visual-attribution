import json
import numpy as np
import torch
import itertools
import pytablewriter

import viz
import utils
from create_explainer import get_explainer
from preprocess import get_preprocess
from explainer.sparse import SparseExplainer

import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import pylab as P


def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
        P.axis('off')

    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)


# def ShowDivergingImage(grad, title='', percentile=99, ax=None):
#     if ax is None:
#         fig, ax = P.subplots()
#     else:
#         fig = ax.figure
#
#     P.axis('off')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     im = ax.imshow(grad, cmap=P.cm.coolwarm, vmin=-1, vmax=1)
#     fig.colorbar(im, cax=cax, orientation='vertical')
#     P.title(title)
#
#
# def LoadImage(file_path):
#     im = PIL.Image.open(file_path)
#     im = np.asarray(im)
#     return im / 127.5 - 1.0


def VisualizeImageGrayscale(image_3d, percentile=99):
    """Returns a 3D tensor as a grayscale 2D tensor.  This method sums a 3D
    tensor across the absolute value of axis=2, and then clips values at a
    given percentile.
    """
    image_3d = np.abs(image_3d.squeeze())
    image_2d = torch.sum(image_3d, dim=0)

    image_2d = image_2d.numpy()
    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    print(vmax)
    print(vmin)

    return torch.from_numpy(np.clip((image_2d - vmin) / (vmax - vmin), 0, 1))


def lambda_l1_n_iterations():
    image_path = 'images/tricycle.png'
    baseline_path = 'images/tricycle_{}.png'
    output_path = 'images/lambda_l1_n_iterations{}.png'
    model_name = 'resnet50'
    method_name = 'sparse'
    show_style = 'imshow'
    raw_img = viz.pil_loader(image_path)
    transf = get_preprocess(model_name, method_name)
    model = utils.load_model(model_name)
    model.cuda()

    lambda_l1s = [1, 1e2, 1e3, 1e4, 1e5, 1e6]
    n_iterations = list(range(1, 15))
    all_configs = list(itertools.product(lambda_l1s, n_iterations))
    all_configs = [{'lambda_l1': ll1, 'n_iterations': n_iter}
                   for ll1, n_iter in all_configs]

    for cfg_id, cfg in enumerate(all_configs):
        explainer = SparseExplainer(model, **cfg)

        inp = transf(raw_img)
        inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)

        # target = torch.LongTensor([image_class]).cuda()
        target = None
        saliency = explainer.explain(inp, target)
        saliency = VisualizeImageGrayscale(saliency)
        saliency = saliency.cpu().numpy()

        if show_style == 'camshow':
            saliency = utils.upsample(np.expand_dims(saliency, axis=0),
                                      (raw_img.height, raw_img.width))
            viz.plot_cam(saliency, raw_img, 'jet', alpha=0.5)
        else:
            plt.imshow(saliency, cmap=P.cm.gray, vmin=0, vmax=1)

        plt.savefig(output_path.format(cfg_id))
        all_configs[cfg_id]['output_path'] = output_path.format(cfg_id)

    baselines = ['vanilla_grad', 'grad_x_input', 'saliency', 'gradcam']
    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = "baselines"
    writer.header_list = ['input'] + baselines
    writer.value_matrix = []
    row = ['![]({})'.format(image_path)]
    row += ['![]({})'.format(baseline_path.format(x)) for x in baselines]
    writer.value_matrix.append(row)
    with open('lambda_l1_n_iterations.md', 'w') as f:
        writer.stream = f
        writer.write_table()

    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = "lambda_l1_n_iterations"
    writer.header_list = [''] + ['n_iter: {}'.format(x) for x in n_iterations]
    writer.value_matrix = []
    for i, lambda_l1 in enumerate(lambda_l1s):
        row = ['lambda l1: {}'.format(lambda_l1)]
        for j, n_iter in enumerate(n_iterations):
            cfg = all_configs[i * len(lambda_l2s) + j]
            # row.append('<img src="{}">'.format(cfg['output_path']))
            row.append('![]({})'.format(cfg['output_path']))
        writer.value_matrix.append(row)
    with open('lambda_l1_n_iterations.md', 'a') as f:
        writer.stream = f
        writer.write_table()


def lambda_l1_l2():
    images = ['8', 'beach', 'elephant',  'obelisk',
    'bird', 'fox', 'tricycle', 'utensil']
    for image in images:
        print(image)
        if not os.path.exists('images/' + image):
            os.makedirs('images/' + image)
        image_path = 'images/' + image + '/' + image + '.png'
        baseline_path = 'images/' + image + '/' + image + '_{}.png'
        output_path = 'images/' + image + '/lambda_l1_l2_{}.png'
        model_name = 'resnet50'
        method_name = 'sparse'
        show_style = 'imshow'
        raw_img = viz.pil_loader(image_path)
        transf = get_preprocess(model_name, method_name)
        
        model = utils.load_model(model_name)
        model.cuda()

        lambda_l1s = [1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
        lambda_l2s = [0, 1, 1e2, 1e3, 1e4, 1e5, 1e6]       
        lambda_l1s = [1e2, 1e3]
        lambda_l2s = [0, 1]
        all_configs = list(itertools.product(lambda_l1s, lambda_l2s))
        all_configs = [{'lambda_l1': ll1, 'lambda_l2': ll2}
                       for ll1, ll2 in all_configs]

        for cfg_id, cfg in enumerate(all_configs):
            explainer = SparseExplainer(model, **cfg)
            inp = transf(raw_img)
            inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)

            # target = torch.LongTensor([image_class]).cuda()
            target = None
            saliency = explainer.explain(inp, target)
            saliency = VisualizeImageGrayscale(saliency)
            saliency = saliency.cpu().numpy()

            if show_style == 'camshow':
                saliency = utils.upsample(np.expand_dims(saliency, axis=0),
                                          (raw_img.height, raw_img.width))
                viz.plot_cam(saliency, raw_img, 'jet', alpha=0.5)
            else:
                plt.imshow(saliency, cmap=P.cm.gray, vmin=0, vmax=1)

            plt.savefig(output_path.format(cfg_id))
            all_configs[cfg_id]['output_path'] = output_path.format(cfg_id)

        baselines = ['vanilla_grad', 'grad_x_input', 'saliency', 'gradcam']
        writer = pytablewriter.MarkdownTableWriter()
        writer.table_name = "baselines"
        writer.header_list = ['input'] + baselines
        writer.value_matrix = []
        row = ['![]({})'.format(image_path)]
        row += ['![]({})'.format(baseline_path.format(x)) for x in baselines]
        writer.value_matrix.append(row)
        with open('images/' + image + '/lambda_l1_l2.md', 'w') as f:
            writer.stream = f
            writer.write_table()

        writer = pytablewriter.MarkdownTableWriter()
        writer.table_name = image + "_lambda_l1_l2"
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
        with open('images/' + image + '/lambda_l1_l2.md', 'a') as f:
            writer.stream = f
            writer.write_table()
        plt.clf()

    def baselines():
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

        image_path = 'images/tricycle.png'
        raw_img = viz.pil_loader(image_path)

        all_saliency_maps = []
        for model_name, method_name, _, kwargs in model_methods:
            transf = get_preprocess(model_name, method_name)
            model = utils.load_model(model_name)
            model.cuda()
            explainer = get_explainer(model, method_name, kwargs)

            inp = transf(raw_img)
            inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)

            # target = torch.LongTensor([image_class]).cuda()
            target = None
            saliency, loss_history = explainer.explain(inp, target, return_loss=True)
            print(loss_history)
            saliency = utils.upsample(saliency, (raw_img.height, raw_img.width))
            all_saliency_maps.append(saliency.cpu().numpy())

        for i, saliency in enumerate(all_saliency_maps):
            model_name, method_name, show_style, extra_args = model_methods[i]
            print(method_name)
            if show_style == 'camshow':
                viz.plot_cam(np.abs(saliency).max(axis=1).squeeze(),
                             raw_img, 'jet', alpha=0.5)
            else:
                if model_name == 'googlenet' or method_name == 'pattern_net':
                    saliency = saliency.squeeze()[::-1].transpose(1, 2, 0)
                else:
                    saliency = saliency.squeeze().transpose(1, 2, 0)
                saliency -= saliency.min()
                saliency /= (saliency.max() + 1e-20)
                plt.imshow(saliency, cmap='gray')

            plt.axis('off')
            output_path = 'images/tricycle_{}.png'.format(method_name)
            plt.savefig(output_path)


def main():
    default_methods = [
        #['resnet50', 'vanilla_grad', 'imshow', None],
        #['resnet50', 'sparse_guided_backprop', 'imshow', None],
        ['resnet50', 'sparse', 'imshow', None],
        # ['resnet50', 'grad_x_input', 'imshow', None],
        # ['resnet50', 'saliency', 'imshow', None],
        # ['resnet50', 'sparse_integrate_grad', 'imshow', None],
        # ['resnet50', 'deconv', 'imshow', None],
        #['resnet50', 'guided_backprop', 'imshow', None],
        #['resnet50', 'smooth_grad', 'imshow', None],
        #['resnet50', 'sparse_smooth_grad', 'imshow', None],
        # ['resnet50', 'gradcam', 'camshow', None],
        # ['resnet50', 'excitation_backprop', 'camshow', None],
        # ['resnet50', 'contrastive_excitation_backprop', 'camshow', None],
        # ['vgg16', 'pattern_net', 'imshow', None],
        # ['vgg16', 'pattern_lrp', 'camshow', None],
        # ['resnet50', 'real_time_saliency', 'camshow', None],
        ]
    '''
    sparse_methods = [
        # ['resnet50', 'sparse', 'camshow',
        #     {'hessian_coefficient': 0, 'lambda_l1': 0, 'lambda_l2': 0}],
        # ['resnet50', 'sparse', 'camshow',
        #     {'hessian_coefficient': 0, 'lambda_l1': 0, 'lambda_l2': 1e3}],
        # ['resnet50', 'sparse', 'camshow',
        #     {'hessian_coefficient': 0, 'lambda_l1': 0, 'lambda_l2': 1e5}],
        # ['resnet50', 'sparse', 'camshow',
        #     {'hessian_coefficient': 0, 'lambda_l1': 1e3, 'lambda_l2': 1e5}],
        # ['resnet50', 'sparse', 'camshow',
        #     {'hessian_coefficient': 0, 'lambda_l1': 1e4, 'lambda_l2': 1e5}],
        ['resnet50', 'sparse', 'imshow',
            {'hessian_coefficient': 1, 'lambda_l1': 5e4, 'lambda_l2': 0, 'n_iterations': 1}],
        ['resnet50', 'sparse', 'imshow',
            {'hessian_coefficient': 1, 'lambda_l1': 5e4, 'lambda_l2': 0, 'n_iterations': 2}],
        ['resnet50', 'sparse', 'imshow',
            {'hessian_coefficient': 1, 'lambda_l1': 5e4, 'lambda_l2': 0, 'n_iterations': 3}],
        ['resnet50', 'sparse', 'imshow',
            {'hessian_coefficient': 1, 'lambda_l1': 5e4, 'lambda_l2': 0, 'n_iterations': 4}],
        ['resnet50', 'sparse', 'imshow',
            {'hessian_coefficient': 1, 'lambda_l1': 5e4, 'lambda_l2': 0, 'n_iterations': 5}],
        ['resnet50', 'sparse', 'imshow',
            {'hessian_coefficient': 1, 'lambda_l1': 5e4, 'lambda_l2': 0, 'n_iterations': 6}],
        ['resnet50', 'sparse', 'imshow',
            {'hessian_coefficient': 1, 'lambda_l1': 5e4, 'lambda_l2': 0, 'n_iterations': 7}],
        ['resnet50', 'sparse', 'imshow',
            {'hessian_coefficient': 1, 'lambda_l1': 5e4, 'lambda_l2': 0, 'n_iterations': 8}],
        ['resnet50', 'sparse', 'imshow',
            {'hessian_coefficient': 1, 'lambda_l1': 5e4, 'lambda_l2': 0, 'n_iterations': 9}],
        ['resnet50', 'sparse', 'imshow',
            {'hessian_coefficient': 1, 'lambda_l1': 5e4, 'lambda_l2': 0, 'n_iterations': 10}],
        # ['resnet50', 'sparse', 'camshow',
        #     {'hessian_coefficient': 1, 'lambda_l1': 5e4, 'lambda_l2': 0}],
        # ['resnet50', 'sparse_smooth_grad', 'imshow', None],
        # ['resnet50', 'sparse', 'camshow',
        #     {'hessian_coefficient': 1, 'lambda_l1': 1e4, 'lambda_l2': 1e5}],
        # ['resnet50', 'sparse', 'camshow',
        #     {'hessian_coefficient': 1, 'lambda_l1': 5e3, 'lambda_l2': 1e5}],
        # ['resnet50', 'sparse', 'camshow',
        #     {'hessian_coefficient': 1, 'lambda_l1': 8e3, 'lambda_l2': 1e5}],
    ]
    '''

    model_methods = default_methods  # + sparse_methods

    image_path = 'images/tricycle.png'
    raw_img = viz.pil_loader(image_path)

    all_saliency_maps = []
    for model_name, method_name, _, kwargs in model_methods:
        print(method_name)
        transf = get_preprocess(model_name, method_name)
        model = utils.load_model(model_name)
        model.cuda()
        explainer = get_explainer(model, method_name, kwargs)

        inp = transf(raw_img)
        inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)

        # target = torch.LongTensor([image_class]).cuda()
        target = None
        saliency = explainer.explain(inp, target)
        saliency = VisualizeImageGrayscale(saliency)
        # saliency = utils.upsample(saliency, (raw_img.height, raw_img.width))

        all_saliency_maps.append(saliency.cpu().numpy())

    plt.figure(figsize=(25, 15))
    plt.subplot(3, 5, 1)
    plt.imshow(raw_img)
    plt.axis('off')
    plt.title('Tricycle')
    for i, saliency in enumerate(all_saliency_maps):
        model_name, method_name, show_style, extra_args = model_methods[i]
        plt.subplot(3, 5, i + 2 + i // 4)
        # if show_style == 'camshow':
        #     viz.plot_cam(np.abs(saliency).max(axis=1).squeeze(),
        #                  raw_img, 'jet', alpha=0.5)
        if show_style == 'camshow':
            viz.plot_cam(utils.upsample(np.expand_dims(saliency, axis=0),
                                        (raw_img.height, raw_img.width)),
                         raw_img, 'jet', alpha=0.5)
        # else:
        #     if model_name == 'googlenet' or method_name == 'pattern_net':
        #         saliency = saliency.squeeze()[::-1].transpose(1, 2, 0)
        #     else:
        #         saliency = saliency.squeeze().transpose(1, 2, 0)
        #     saliency -= saliency.min()
        #     saliency /= (saliency.max() + 1e-20)
        else:
            plt.imshow(saliency, cmap=P.cm.gray, vmin=0, vmax=1)

        plt.axis('off')
        if method_name == 'excitation_backprop':
            plt.title('Exc_bp')
        elif method_name == 'contrastive_excitation_backprop':
            plt.title('CExc_bp')
        elif extra_args is not None:
            plt.title(json.dumps(extra_args))
        else:
            plt.title(method_name)

    plt.tight_layout()
    plt.savefig('images/tusker_saliency.png')


if __name__ == '__main__':
    main()
    #baselines()
    #lambda_l1_l2()
    #lambda_l1_n_iterations()
    lambda_l1_l2()