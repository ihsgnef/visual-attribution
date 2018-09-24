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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import pylab as P

def gray_out(model, X, protected=None):
        protected = np.repeat(protected[:, np.newaxis, :, :], 3, axis=1)
        X = X.cpu().data.numpy()
        X = X * -(protected - 1)    # zero out the important region
        X = X + (protected * 0.5) # set x to 0.5 values
        X = Variable(torch.from_numpy(X).cuda(), requires_grad = True).float()
        # change x at the unprotected regions to the mean values (0.4914, 0.4822, 0.4465)
        return X

# class LinfPGDAttack(object):
#     def __init__(self, model=None, epsilon=0.3, k=40, a=0.01, 
#         random_start=True):
#         """
#         Attack parameter initialization. The attack performs k steps of
#         size a, while always staying within epsilon from the initial
#         point.
#         https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
#         """
#         self.model = model
#         self.epsilon = epsilon
#         self.k = k
#         self.a = a
#         self.rand = random_start
#         self.loss_fn = nn.CrossEntropyLoss()

#     def perturb(self, X_nat, y):
#         """
#         Given examples (X_nat, y), returns adversarial
#         examples within epsilon of X_nat in l_infinity norm.
#         """
#         if self.rand:
#             X = X_nat + np.random.uniform(-self.epsilon, self.epsilon,
#                 X_nat.shape).astype('float32')
#         else:
#             X = np.copy(X_nat)

#         for i in range(self.k):
#             X_var = to_var(torch.from_numpy(X), requires_grad=True)
#             y_var = to_var(torch.LongTensor(y))

#             scores = self.model(X_var)
#             loss = self.loss_fn(scores, y_var)
#             loss.backward()
#             grad = X_var.grad.data.cpu().numpy()

#             X += self.a * np.sign(grad)

#             X = np.clip(X, X_nat - self.epsilon, X_nat + self.epsilon)
#             X = np.clip(X, 0, 1) # ensure valid pixel range

#         return X

def iterative_perturb(model, X_nat, y=None, epsilon=2.0/255.0, protected=None):
    protected = np.repeat(protected[:, np.newaxis, :, :], 3, axis=1)
    random_perturb = protected * np.random.uniform(-epsilon, epsilon,
                X_nat.shape) 
    perturbed_X = Variable(X_nat.data + torch.from_numpy(random_perturb).cuda().float(), requires_grad=True)
    if y is None:
        y = model(X_nat).max(1)[1]

    for i in range(40):
        output = model(perturbed_X)
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad_sign = perturbed_X.grad.data.cpu().sign().numpy()
        perturbed_X  = perturbed_X.data.cpu().numpy() + .001 * grad_sign * protected
        perturbed_X = np.clip(perturbed_X, X_nat.data.cpu().numpy() - epsilon, X_nat.data.cpu().numpy() + epsilon)
        perturbed_X = np.clip(perturbed_X, 0, 1)
        perturbed_X = Variable(torch.from_numpy(perturbed_X).cuda().float(), requires_grad=True)

    return perturbed_X


def single_step_perturb(model, X, y=None, epsilon=2.0/255.0, protected=None):
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

def attackImportant(saliency, cutoff = 10):
    if cutoff == 0:
        return np.zeros_like(saliency)
    if cutoff == 100:
        return np.ones_like(saliency)

    batch_size, height, width = saliency.shape
    saliency = np.abs(saliency)
    new_saliency = []
    for i in range(batch_size):
        sal = saliency[i].copy()
        protected_percentile = np.percentile(sal, cutoff)
        k = int(cutoff / 100 * height * width) - 1        
        sal = np.reshape(sal,(height*width))
        bottom_k = np.argpartition(sal, -k)
        sal = np.zeros_like(sal).astype(int)
        for idx in bottom_k[-k:]:
            sal[idx] = 1
        sal = np.reshape(sal, (height,width))
        new_saliency.append(sal)
    new_saliency = np.stack(new_saliency)
    assert new_saliency.shape == (batch_size, height, width)
    return new_saliency


def attackUnImportant(saliency, cutoff = 10):
    if cutoff == 0:
        return np.zeros_like(saliency)
    if cutoff == 100:
        return np.ones_like(saliency)

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
    transf = get_preprocess('resnet18', 'sparse',dataset)
    #model = utils.load_model('cifar50')
    model = utils.load_model('resnet18')
    model.cuda()
    model.eval()

    explainers = [
        ('Vanilla', VanillaGradExplainer()),
        ('Random', None),
        ('SmoothGrad', SmoothGradExplainer()),
        #('Sparse', SparseExplainer()),
        #('Tuned_Sparse', LambdaTunerExplainer()),
        #('Random', None),
        #('IntegratedGrad', IntegrateGradExplainer()),
    ]


    cutoff_scores = dict()
    for method_name, explainer in explainers:
        cutoff_scores[method_name] = [0] * 11

    cutoffs = [10]#0,1,2,3,4,5,6,7,8,9,10]#,20,30]#[0,10,20,30,40,50,60,70,80,90,100] # percentage adversary can see
    num_images = 1#16#32

    batch_size = 1#16

    batches = utils.load_data(batch_size=batch_size, num_images = num_images, transf=transf, dataset=dataset)
    for batch in batches:
        inputs = Variable(batch.cuda(), requires_grad=True)
        forward_pass = model(inputs)
        original_prediction = forward_pass.max(1, keepdim=True)[1]
        original_confidence = F.softmax(forward_pass, dim=1)
        #confidence_for_prediction = original_confidence[original_prediction]#.max(1, keepdim=True)
        confidence_for_class = original_confidence.cpu().data.numpy()[0][original_prediction.cpu().data.numpy()][0][0]

        raw_img = batch.cpu().numpy()[0]#viz.pil_loader(batch.cpu().numpy()[0])
        all_saliency_maps = []                

        for method_name, explainer in explainers:
            if method_name == "Random":
                saliency = torch.from_numpy(np.random.rand(*inputs.shape)).float()
            else:
                saliency = explainer.explain(model, copy.deepcopy(inputs).data)
            saliency = viz.VisualizeImageGrayscale(saliency.cpu())

            for cutoff in cutoffs:
                protected_region = attackImportant(saliency.cpu().numpy(), cutoff=cutoff)

                #adversarial_image = single_step_perturb(model, copy.deepcopy(inputs), protected = protected_region)                                          
                # adversarial_prediction = model(adversarial_image).max(1, keepdim=True)[1]
                # correct = original_prediction.eq(adversarial_prediction).sum().cpu().data.numpy()
                
                gray_image = gray_out(model, copy.deepcopy(inputs), protected = protected_region)
                gray_confidence = F.softmax(model(gray_image), dim=1)
                gray_confidence_for_class = gray_confidence.cpu().data.numpy()[0][original_prediction.cpu().data.numpy()][0][0]
                # print(confidence_for_class)
                # print(gray_confidence_for_class)
                # print(original_prediction[0])
                # print(gray_confidence.max(1, keepdim=True))
                # print()
                correct = confidence_for_class - gray_confidence_for_class

                # adversarial_image = iterative_perturb(model, copy.deepcopy(inputs), protected = protected_region)
                # adversarial_prediction = model(adversarial_image).max(1, keepdim=True)[1]
                # correct = original_prediction.eq(adversarial_prediction).sum().cpu().data.numpy()

                cutoff_scores[method_name][int(cutoff/10)] += float(correct / batch_size)
                            
                protected_region = np.repeat(protected_region[:, np.newaxis, :, :], 3, axis=1)                
                all_saliency_maps.append(batch.cpu().numpy()[0] * (1 - protected_region))# + (128 * protected_region))

        plt.figure(figsize=(25, 15))
        plt.subplot(3, 5, 1)
        
        raw_img = np.swapaxes(raw_img, 1,2)
        plt.imshow(np.swapaxes(raw_img, 0,2))
        plt.axis('off')
        plt.title('Dog')
        for i, saliency in enumerate(all_saliency_maps):                    
            plt.subplot(3, 5, i + 2 + i // 4)                                
            saliency = saliency[0]
            saliency = np.swapaxes(saliency, 1,2)
            plt.imshow(np.swapaxes(saliency, 0,2))#, cmap=P.cm.gray, vmin=0, vmax=1)

            plt.axis('off')        
            #plt.title("")
        plt.tight_layout()
        plt.savefig('output/protected_attack.png')



    for cutoff in cutoffs:
        print("Adversary Can Modify: ", cutoff)
        for method_name, explainer in explainers:
            print(method_name, cutoff_scores[method_name][int(cutoff/10)] / (num_images / batch_size))

        # with open("protected_results.txt", "a") as text_file:

        #     text_file.write('\n' + str(cutoff) + '\n' +'\n')
        #     for method_name in all_scores:
        #         accuracy = all_scores[method_name] / float(num_images / batch_size)
        #         print(method_name, accuracy)
        #         text_file.write(str(method_name) + '\n')
        #         text_file.write(str(accuracy) + '\n')

