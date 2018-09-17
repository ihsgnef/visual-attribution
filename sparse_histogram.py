import numpy as np
import torch
import viz
import utils
from create_explainer import get_explainer
from preprocess import get_preprocess
import copy
from collections import Iterable
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import glob
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import pylab as P

batch_size = 1
if __name__ == '__main__':    
    image_path = '/fs/imageNet/imagenet/ILSVRC_val/**/*.JPEG'    
    image_files = list(glob.iglob(image_path, recursive=True))
    np.random.seed(0)
    np.random.shuffle(image_files)
    image_files = image_files[:100]
    indices = list(range(0, len(image_files), batch_size))
    model = utils.load_model('resnet18')
    model.cuda()
    model.eval()
    sparse_args = {
        'lambda_t1': 1,
        'lambda_t2': 0,
        'lambda_l1': 1e2,
        'lambda_l2': 1e4,
        'n_iterations': 10,
        'optim': 'sgd',
        'lr': 1e-1,
    }
    
    saliency_histo_list_grad = []
    saliency_histo_list_sparse = []
    for batch_idx, start in enumerate(indices):
        batch = image_files[start: start + batch_size]
        raw_images = [viz.pil_loader(x) for x in batch]
        transf = get_preprocess('resnet50', 'sparse')
        inputs = torch.stack([transf(x) for x in raw_images])
        inputs = Variable(inputs.cuda(), requires_grad=True)
    
        explainer = get_explainer(model, 'vanilla_grad', None)
        saliency = explainer.explain(copy.deepcopy(inputs), None)
        saliency = viz.VisualizeImageGrayscale(saliency)       
        saliency_histo_list_grad.extend(saliency.cpu().numpy()[0].ravel())

        explainer = get_explainer(model, 'sparse', sparse_args)
        saliency = explainer.explain(copy.deepcopy(inputs), None)
        saliency = viz.VisualizeImageGrayscale(saliency)       
        saliency_histo_list_sparse.extend(saliency.cpu().numpy()[0].ravel())        
    
    n_bins = 40
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)    
    axs[0].hist(saliency_histo_list_grad, bins=n_bins)
    axs[1].hist(saliency_histo_list_sparse, bins=n_bins)
    plt.savefig('output/histogram.png')
    print(len(saliency_histo_list_sparse))
    print(len(saliency_histo_list_grad))

