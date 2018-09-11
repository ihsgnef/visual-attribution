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

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import pylab as P


import copy
from collections import Iterable
from scipy.stats import truncnorm
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def perturb(model, X, y=None, epsilon=0.3, protected=None):         
    
    #X_var = Variable(torch.from_numpy(X).cuda(), requires_grad=True, volatile=False)
    #y_var = Variable(torch.LongTensor(y).cuda(), requires_grad=False, volatile=False)

    #output = model(X_var)
    output = model(X)
    if y is None:
        y = output.max(1)[1]
    loss = F.cross_entropy(output, y)        
    loss.backward()
    #grad_sign = X_var.grad.data.cpu().sign().numpy()
    grad_sign = X.grad.data.cpu().sign().numpy()

    perturbed_X = X.data.cpu().numpy() + epsilon * grad_sign
    perturbed_X = np.clip(perturbed_X, 0, 1)    
    X = Variable(torch.from_numpy(perturbed_X).cuda(), requires_grad = True)    
    return X

def getProtectedRegion(saliency, cutoff = 0.25):        
    return np.abs(saliency) < cutoff*np.abs(saliency).max() # note for paper we do absolute value when computing it

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
        P.axis('off')

    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)

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

def main():
    model_methods = [
        ['resnet50', 'vanilla_grad', 'imshow', None]]	    

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
        if method_name == 'googlenet':  # swap channel due to caffe weights
            inp_copy = inp.clone()
            inp[0] = inp_copy[2]
            inp[2] = inp_copy[0]
        inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)
                
        target = None          
        saliency = explainer.explain(inp, target)
        saliency = VisualizeImageGrayscale(saliency)        
        all_saliency_maps.append(saliency.cpu().numpy())

        protected_region = getProtectedRegion(saliency.cpu().numpy())                            
        all_saliency_maps.append(protected_region) # plot protected region

        adversarial_image = perturb(model, inp, protected = None)                
        original_prediction = model(inp).max(1)[1]
        adversarial_prediction = model(adversarial_image).max(1)[1]

        print(original_prediction)
        print(adversarial_prediction)
        # if original_prediction == adversarial_prediction:
        #     print("Correct!")
        # else:
        #     print("Incorrect!")
        
        adversarial_saliency = explainer.explain(adversarial_image, adversarial_prediction.cuda()) # explain using new prediction
        adversarial_saliency = VisualizeImageGrayscale(adversarial_saliency)        
        all_saliency_maps.append(adversarial_saliency.cpu().numpy())


    model_methods = [
        ['resnet50', 'vanilla_grad', 'imshow', None],
        ['resnet50', 'protected', 'imshow', None],
        ['resnet50', 'adversary', 'imshow', None],
        ]       

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
