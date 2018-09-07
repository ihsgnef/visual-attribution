import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import PIL.Image
from matplotlib import pylab as P

import viz
import utils
from create_explainer import get_explainer
from preprocess import get_preprocess

def ShowGrayscaleImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')

  P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
  P.title(title)

# def ShowDivergingImage(grad, title='', percentile=99, ax=None):  
#   if ax is None:
#     fig, ax = P.subplots()
#   else:
#     fig = ax.figure
  
#   P.axis('off')
#   divider = make_axes_locatable(ax)
#   cax = divider.append_axes('right', size='5%', pad=0.05)
#   im = ax.imshow(grad, cmap=P.cm.coolwarm, vmin=-1, vmax=1)
#   fig.colorbar(im, cax=cax, orientation='vertical')
#   P.title(title)

# def LoadImage(file_path):
#   im = PIL.Image.open(file_path)
#   im = np.asarray(im)
#   return im / 127.5 - 1.0


def VisualizeImageGrayscale(image_3d, percentile=99):
  r"""Returns a 3D tensor as a grayscale 2D tensor.
  This method sums a 3D tensor across the absolute value of axis=2, and then
  clips values at a given percentile.
  """  
  image_3d = np.abs(image_3d.squeeze())
  image_2d = torch.sum(image_3d, dim=0)

  image_2d = image_2d.numpy()
  vmax = np.percentile(image_2d, percentile)
  vmin = np.min(image_2d)

  return torch.from_numpy(np.clip((image_2d - vmin) / (vmax - vmin), 0, 1))

def main():
    model_methods = [
        ['resnet50', 'vanilla_grad', 'imshow', None],
        # ['resnet50', 'grad_x_input', 'imshow', None],
        # ['resnet50', 'saliency', 'imshow', None],
        #['resnet50', 'sparse_integrate_grad', 'imshow', None],
        # ['resnet50', 'deconv', 'imshow', None],
        # ['resnet50', 'guided_backprop', 'imshow', None],
        # ['resnet50', 'gradcam', 'camshow', None],
        # ['resnet50', 'excitation_backprop', 'camshow', None],
        # ['resnet50', 'contrastive_excitation_backprop', 'camshow', None],
        # ['vgg16', 'pattern_net', 'imshow', None],
        # ['vgg16', 'pattern_lrp', 'camshow', None],
        #['resnet50', 'real_time_saliency', 'camshow', None],
        #['resnet50', 'sparse', 'camshow',
        #    {'hessian_coefficient': 0, 'lambda_l1': 0, 'lambda_l2': 0}],
        #['resnet50', 'sparse', 'camshow',
        #    {'hessian_coefficient': 0, 'lambda_l1': 0, 'lambda_l2': 1e3}],
        #['resnet50', 'sparse', 'camshow',
        #    {'hessian_coefficient': 0, 'lambda_l1': 0, 'lambda_l2': 1e5}],
        #['resnet50', 'sparse', 'camshow',
        #    {'hessian_coefficient': 0, 'lambda_l1': 1e3, 'lambda_l2': 1e5}],
        #['resnet50', 'sparse', 'camshow',
        #    {'hessian_coefficient': 0, 'lambda_l1': 1e4, 'lambda_l2': 1e5}],
        #['resnet50', 'sparse', 'imshow',
        #    {'hessian_coefficient': 1, 'lambda_l1': 5e4, 'lambda_l2': 0}],                    
        ['resnet50', 'sparse', 'camshow',
            {'hessian_coefficient': 1, 'lambda_l1': 5e4, 'lambda_l2': 0}],            
        #['resnet50', 'sparse_smooth_grad', 'imshow', None],
        #['resnet50', 'sparse', 'camshow',
        #    {'hessian_coefficient': 1, 'lambda_l1': 1e4, 'lambda_l2': 1e5}],
        #['resnet50', 'sparse', 'camshow',
        #    {'hessian_coefficient': 1, 'lambda_l1': 5e3, 'lambda_l2': 1e5}],
        #['resnet50', 'sparse', 'camshow',
        #    {'hessian_coefficient': 1, 'lambda_l1': 8e3, 'lambda_l2': 1e5}],
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
        if method_name == 'googlenet':  # swap channel due to caffe weights
            inp_copy = inp.clone()
            inp[0] = inp_copy[2]
            inp[2] = inp_copy[0]
        inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)

        #target = torch.LongTensor([image_class]).cuda()
        saliency = explainer.explain(inp, None)#target)
        saliency = VisualizeImageGrayscale(saliency)
        

        #saliency = utils.upsample(saliency, (raw_img.height, raw_img.width))

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
            viz.plot_cam(utils.upsample(np.expand_dims(saliency, axis=0), (raw_img.height, raw_img.width)),
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
