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
import torchvision.transforms as transforms

import torchvision.transforms.functional as F_trans

from torchvision.models.inception import inception_v3

import os
import glob

# inv_normalize = transforms.Compose([         
#     transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
#         std=[1/0.229, 1/0.224, 1/0.255]),
#     transforms.Scale((299, 299)),
#     # transforms.ToPILImage(),
#     ])

def perturb(model, X, y=None, epsilon=2 / 255, protected=None):         
    
    #X_var = Variable(torch.from_numpy(X).cuda(), requires_grad=True, volatile=False)
    #y_var = Variable(torch.LongTensor(y).cuda(), requires_grad=False, volatile=False)

    #output = model(X_var)
    ################################output = model(transf(X))
    output = model(X)
    if y is None:
        y = output.max(1)[1]
    loss = F.cross_entropy(output, y)        
    loss.backward()
    #grad_sign = X_var.grad.data.cpu().sign().numpy()
    grad_sign = X.grad.data.cpu().sign().numpy()
    #################################################################### TODO, we use different maskes
    protected = np.repeat(protected[np.newaxis, :, :], 3, axis=0)
    grad_sign = grad_sign * protected     
    
    perturbed_X = X.data.cpu().numpy() + epsilon * grad_sign                 
    perturbed_X = np.clip(perturbed_X, 0, 1)    
    X = Variable(torch.from_numpy(perturbed_X).cuda(), requires_grad = True)    
    return X

def getProtectedRegion(saliency, cutoff = 0.05):                
    assert saliency == nps.abs(saliency)    ### shouldn't be different because we normalize beforehand
    saliency = np.abs(saliency)
    protected_percentile = np.percentile(saliency, cutoff)
    return saliency < protected_percentile # note for paper we do absolute value when computing it. Is that a good decision?    

def VisualizeImageGrayscale(image_3d, percentile=99):
    image_3d = np.abs(image_3d.squeeze())
    image_2d = torch.sum(image_3d, dim=0)

    image_2d = image_2d.numpy()
    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    return torch.from_numpy(np.clip((image_2d - vmin) / (vmax - vmin), 0, 1))

def main():
    model = utils.load_model('resnet50')
    model.cuda()
    vanilla_grad_explainer = get_explainer(model, 'vanilla_grad', None)
    random_explainer = get_explainer(model, 'random', None)
    sparse_explainer = get_explainer(model, 'sparse', None)
    explainers = [vanilla_grad_explainer, random_explainer]#, sparse_explainer]
    explainers_correct = [0, 0]
    transf = transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor(),        
    ])

    target = None          
    image_path = '/fs/imageNet/imagenet/ILSVRC_val/'
    
    num_total = 0

    cutoffs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for current_cutoff in cutoffs:
        for filename in glob.iglob(image_path + '**/*.JPEG', recursive=True):
            num_total = num_total + 1
            if (num_total > 50):
                continue
            raw_img = viz.pil_loader(filename)
            inp = transf(raw_img)            ######################################### TODO ERIC KNOWS THIS        
            inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)
            for idx, explainer in enumerate(explainers):                        
                saliency = explainer.explain(inp, target)
                saliency = VisualizeImageGrayscale(saliency)        
                protected_region = getProtectedRegion(saliency.cpu().numpy(), cutoff=current_cutoff)
                adversarial_image = perturb(model, inp, protected = protected_region)                
                
                original_prediction = model(inp).max(1)[1]        
                adversarial_prediction = model(adversarial_image).max(1)[1]            
                if (original_prediction.data.cpu().numpy()[0] == adversarial_prediction.data.cpu().numpy()[0]):
                    explainers_correct[idx] = explainers_correct[idx] + 1
    	
        print("Current Cutoff: ", current_cutoff)
        print("Epsilon: ", epsilon)
        for idx, explainer_correct in enumerate(explainers_correct):
            print("Method: ", explainers[idx], "Protected Accuracy: ", float(explainer_correct) / 50)#float(num_total))           
if __name__ == '__main__':
    main()    
