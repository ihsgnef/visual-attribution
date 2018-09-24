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
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
from explainers_redo import SparseExplainer, RobustSparseExplainer, \
    VanillaGradExplainer, IntegrateGradExplainer, SmoothGradExplainer, \
    LambdaTunerExplainer

def perturb(model, X, y=None, epsilon=2.0/255.0, protected=None):
    output = model(X)
    if y is None:
        y = output.max(1)[1]
    loss = F.cross_entropy(output, y)
    loss.backward()
    grad_sign = X.grad.data.cpu().sign().numpy()
    protected = np.repeat(protected[:, np.newaxis, :, :], 3, axis=1)    
    grad_sign = grad_sign * protected
    perturbed_X = X.data.cpu().numpy() + epsilon * grad_sign
    perturbed_X = np.clip(perturbed_X, 0, 1)
    X = Variable(torch.from_numpy(perturbed_X).cuda(), requires_grad = True).float()
    return X

def attackUnImportant(saliency, cutoff = 10):
    if cutoff == 0:
        return np.zeros_like(saliency)
    batch_size, height, width = saliency.shape
    saliency = np.abs(saliency)
    new_saliency = []
    for i in range(batch_size):
        sal = saliency[i].copy()
        protected_percentile = np.percentile(sal, cutoff)
        k = int(cutoff / 100 * height * width) - 1
        sal = np.reshape(sal,(height*width)) 
        bottom_k = np.argpartition(sal, k)
        sal = np.zeros_like(sal).astype(int)
        for idx in bottom_k[:k]:
            sal[idx] = 1        
        sal = np.reshape(sal, (height,width))
        new_saliency.append(sal)
    new_saliency = np.stack(new_saliency)
    assert new_saliency.shape == (batch_size, height, width)
    return new_saliency

if __name__ == '__main__':
    #dataset = 'cifar10'
    dataset = 'imagenet'
    transf = get_preprocess('resnet50', 'sparse',dataset)
    #model = utils.load_model('cifar50')
    model = utils.load_model('resnet50')
    model.cuda()
    model.eval()

    explainers = [
        # ('Sparse', SparseExplainer()),
        # ('Tuned_Sparse', LambdaTunerExplainer()),
        ('Vanilla', VanillaGradExplainer()),
        ('Random', None),
        ('SmoothGrad', SmoothGradExplainer()),
        # ('IntegratedGrad', IntegrateGradExplainer()),
    ]

    cutoff_scores = dict()
    for explainer in explainers:
        cutoff_scores[explainer] = [0] * 11    

    cutoffs = [0,10,20,30,40,50,60,70,80,90,100] # percentage adversary can see
    num_images = 16
    batch_size = 16

    batches = utils.load_data(batch_size=batch_size, num_images = num_images, transf=transf, dataset=dataset)
    for batch in batches:
        inputs = Variable(batch.cuda(), requires_grad=True)    
        original_prediction = model(inputs).max(1, keepdim=True)[1]

        for method_name, explainer in explainers:
            if method_name == "random":
                saliency = torch.from_numpy(np.random.rand(*inputs.shape)).cuda()            
            else:
                saliency = explainer.explain(model, copy.deepcopy(inputs))
            saliency = viz.VisualizeImageGrayscale(saliency.cpu())

            for cutoff in cutoffs:
                protected_region = attackUnImportant(saliency.cpu().numpy(), cutoff=cutoff)
                adversarial_image = perturb(model, copy.deepcopy(inputs), protected = protected_region)            
                adversarial_prediction = model(adversarial_image).max(1, keepdim=True)[1]
                correct = original_prediction.eq(adversarial_prediction).sum().cpu().data.numpy()
                cutoff_scores[explainer][cutoff / 10] += float(correct / batch_size)
    

    for cutoff in cutoffs:
        print("Adversary Can Modify: ", cutoff)
        for explainer in explainers:
            print(explainer, cutoff_scores[explainer][cutoff/10])                    

        # with open("protected_results.txt", "a") as text_file:
            
        #     text_file.write('\n' + str(cutoff) + '\n' +'\n')
        #     for method_name in all_scores:
        #         accuracy = all_scores[method_name] / float(num_images / batch_size)
        #         print(method_name, accuracy)
        #         text_file.write(str(method_name) + '\n')
        #         text_file.write(str(accuracy) + '\n')
