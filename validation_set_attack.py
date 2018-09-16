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


batch_size = 16

# inv_normalize = transforms.Compose([         
#     transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
#         std=[1/0.229, 1/0.224, 1/0.255]),
#     transforms.Scale((299, 299)),
#     # transforms.ToPILImage(),
#     ])

def perturb(model, X, y=None, epsilon=2.0/255.0, protected=None):         
    
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

def attackUnImportant(saliency, cutoff = 0.10):               
    # if cutoff == 0:
    #     protected_percentile = -1  
    batch_size, height, width = saliency.shape  
    saliency = np.abs(saliency)
    new_saliency = []
    for i in range(batch_size):
        sal = saliency[i].copy()
        protected_percentile = np.percentile(sal, cutoff)
        sal = sal <= protected_percentile
        new_saliency.append(sal)
    new_saliency = np.stack(new_saliency)
    assert new_saliency.shape == (batch_size, height, width)

def run_protected(raw_images, cutoff):
    transf = get_preprocess('resnet50', 'sparse')
    model = utils.load_model('resnet50')
    model.eval()
    model.cuda()

    sparse_args = {
        'lambda_t1': 1,
        'lambda_t2': 1,
        'lambda_l1': 0,
        'lambda_l2': 1e4,
        'n_iterations': 10,
        'optim': 'sgd',
        'lr': 1e-1,
    }

    configs = [
        ['sparse', sparse_args],
        ['vanilla_grad', None],
        ['grad_x_input', None],
        ['smooth_grad', None],
        ['integrate_grad', None],
    ]
    
    
    inputs = torch.stack([transf(x) for x in raw_images])
    inputs = Variable(inputs.cuda(), requires_grad=True)
    for method_name, kwargs in configs:
        if method_name == "random":                    
            saliency = torch.from_numpy(np.random.rand(batch_size,3,224,224)).cuda()                      
            saliency = viz.VisualizeImageGrayscale(saliency)
            protected_region = attackUnImportant(saliency.cpu().numpy(), cutoff=current_cutoff)
        else:
            explainer = get_explainer(model, method_name, kwargs)
            saliency = explainer.explain(copy.deepcopy(inputs), None)
            saliency = viz.VisualizeImageGrayscale(saliency)        
            protected_region = attackUnImportant(saliency.cpu().numpy(), cutoff=current_cutoff)

    exit()



        # adversarial_image = perturb(model, copy.deepcopy(inp), protected = protected_region)                        
        # original_prediction = model(inp).max(1)[1]        
        # adversarial_prediction = model(adversarial_image).max(1)[1]           
        # if (original_prediction.data.cpu().numpy()[0] == adversarial_prediction.data.cpu().numpy()[0]):
        #     explainers_correct[idx] = explainers_correct[idx] + 1
    	
        # with open("protected_results.txt", "a") as text_file:
        #     print("Adversary Can Modify: ", current_cutoff)
        #     text_file.write('\n' + str(current_cutoff) + '\n' +'\n')
        #     for idx, explainer_correct in enumerate(explainers_correct):
        #         print(explainer_correct)
        #         print("Method: ", explainers[idx], "Protected Accuracy: ", float(explainer_correct) / num_total)#float(num_total))           
        #         text_file.write(str(explainers[idx]) + '\n')
        #         text_file.write(str(float(explainer_correct) / num_total) + '\n')#float(num_total))   

if __name__ == '__main__':
    cutoffs = [0, 10,20,30,40,50,60,70,80,90,100] # percentage adversary can see    
    for cutoff in cutoffs:
        image_path = '/fs/imageNet/imagenet/ILSVRC_val/**/*.JPEG'
        image_files = list(glob.iglob(image_path, recursive=True))[:64]
        indices = list(range(0, len(image_files), batch_size))
        all_scores = None
        for batch_idx, start in enumerate(indices):
            if batch_idx > 1:
                break
            batch = image_files[start: start + batch_size]
            raw_images = [viz.pil_loader(x) for x in batch]
            scores = run_protected(raw_images, cutoff)
            if all_scores is None:
                all_scores = scores
                continue
            for method_name in scores:
                for attack_name, values in scores[method_name].items():
                    all_scores[method_name][attack_name] += values

        for method_name in all_scores:
            for attack_name, values in all_scores[method_name].items():
                values = list(map(list, zip(*values)))
                print(method_name, attack_name, [np.mean(x) for x in values])
