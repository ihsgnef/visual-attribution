import numpy as np
import torch
import itertools
import pytablewriter

import viz
import utils
from viz import VisualizeImageGrayscale
from create_explainer import get_explainer
from preprocess import get_preprocess
from explainer.sparse import SparseExplainer

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def get_saliency(model, explainer, inp, raw_img,
                 model_name, method_name, viz_style, filename):
    if method_name == 'googlenet':  # swap channel due to caffe weights
        inp_copy = inp.clone()
        inp[0] = inp_copy[2]
        inp[2] = inp_copy[0]
    inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)

    # target = torch.LongTensor([image_class]).cuda()
    target = None
    saliency = explainer.explain(inp, target)
    saliency = utils.upsample(saliency, (raw_img.height, raw_img.width))
    saliency = saliency.cpu().numpy()

    if viz_style == 'camshow':
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
    plt.savefig(filename)


def main():
    input_path = 'examples/tricycle.png'
    model_name = 'resnet50'
    method_name = 'sparse'
    viz_style = 'imshow'
    raw_img = viz.pil_loader(input_path)
    transf = get_preprocess(model_name, method_name)
    model = utils.load_model(model_name)
    model.cuda()
