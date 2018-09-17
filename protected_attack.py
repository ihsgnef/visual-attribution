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

batch_size = 8

# inv_normalize = transforms.Compose([         
#     transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
#         std=[1/0.229, 1/0.224, 1/0.255]),
#     transforms.Scale((299, 299)),
#     # transforms.ToPILImage(),
#     ])

def perturb(model, X, y=None, epsilon=2.0/255.0, protected=None):         
    output = model(X)
    if y is None:
        y = output.max(1)[1]
    loss = F.cross_entropy(output, y)        
    loss.backward()    
    grad_sign = X.grad.data.cpu().sign().numpy()
    #################################################################### TODO, use different masks?
    protected = np.repeat(protected[:, np.newaxis, :, :], 3, axis=1)
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
    return new_saliency

def run_protected(raw_images, cutoff):
    transf = get_preprocess('resnet50', 'sparse')
    model = utils.load_model('resnet50')
    model.cuda()
    model.eval()

    sparse_args = {
        'lambda_t1': 1,
        'lambda_t2': 1,
        'lambda_l1': 1000,
        'lambda_l2': 1e4,
        'n_iterations': 10,
        'optim': 'sgd',
        'lr': 1e-1,
    }

    configs = [
        ['sparse', sparse_args],
        ['vanilla_grad', None],
        #['random', None], 
        #['grad_x_input', None],
        #['smooth_grad', None],
        #['integrate_grad', None],
    ]
        
    inputs = torch.stack([transf(x) for x in raw_images])
    inputs = Variable(inputs.cuda(), requires_grad=True)
    scores = dict()
    for method_name, kwargs in configs:
        if method_name == "random":                    
            saliency = torch.from_numpy(np.random.rand(len(raw_images),3,224,224)).cuda()                      
            saliency = viz.VisualizeImageGrayscale(saliency)
            protected_region = attackUnImportant(saliency.cpu().numpy(), cutoff=cutoff)
        else:
            explainer = get_explainer(model, method_name, kwargs)
            saliency = explainer.explain(copy.deepcopy(inputs), None)
            saliency = viz.VisualizeImageGrayscale(saliency)       
            protected_region = attackUnImportant(saliency.cpu().numpy(), cutoff=cutoff)
    
        adversarial_image = perturb(model, copy.deepcopy(inputs), protected = protected_region)                        
        original_prediction = model(inputs).max(1, keepdim=True)[1]        
        adversarial_prediction = model(adversarial_image).max(1, keepdim=True)[1]        
        correct = original_prediction.eq(adversarial_prediction).sum().cpu().data.numpy()
        scores[method_name] = float(correct / batch_size)
    return scores

if __name__ == '__main__':
    cutoffs = [10,20,30,40,50,60,70,80,90] # percentage adversary can see    
    for cutoff in cutoffs:
        image_path = '/fs/imageNet/imagenet/ILSVRC_val/**/*.JPEG'
        image_files = list(glob.iglob(image_path, recursive=True))
        np.random.seed(0)
        np.random.shuffle(image_files)
        image_files = image_files[:1000]
        indices = list(range(0, len(image_files), batch_size))
        all_scores = None
        for batch_idx, start in enumerate(indices):
            batch = image_files[start: start + batch_size]
            raw_images = [viz.pil_loader(x) for x in batch]
            scores = run_protected(raw_images, cutoff)
            if all_scores is None:
                all_scores = scores
                continue
            for method_name in scores:                
                all_scores[method_name] += scores[method_name]

        with open("protected_results.txt", "a") as text_file:
            print("Adversary Can Modify: ", cutoff)
            text_file.write('\n' + str(cutoff) + '\n' +'\n')
            for method_name in all_scores:
                accuracy = all_scores[method_name] / float(len(image_files) / batch_size)
                print(method_name, accuracy)
                text_file.write(str(method_name) + '\n')
                text_file.write(str(accuracy) + '\n')
