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

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, 2, dim=1, keepdim=True) + 1e-8
    return d

def vat_attack(model, x, ind=None, epsilon=2.0/255.0, protected=None):
        protected = np.repeat(protected[:, np.newaxis, :, :], 3, axis=1)
        n_iterations = 1
        xi = 1e-6

        output = model(x)
        ind = output.max(1)[1]
        
        d = torch.rand(x.shape).sub(0.5).cuda()
        d = _l2_normalize(d)

        for _ in range(n_iterations):
            model.zero_grad()
            d = Variable(xi * d, requires_grad=True)
            pred_hat = model(x + d)            
            adv_loss = F.cross_entropy(pred_hat, ind)
            d_grad, = torch.autograd.grad(adv_loss, d)
            d = _l2_normalize(d_grad.data)        
        d_sign = d.cpu().sign().numpy() * epsilon * protected_region
        perturbed_X = x.data.cpu().numpy() + d_sign 
        perturbed_X = np.clip(perturbed_X, 0, 1)
        return Variable(torch.from_numpy(perturbed_X).cuda(), requires_grad = True).float()

def gray_out(model, X, protected=None):
        protected = np.repeat(protected[:, np.newaxis, :, :], 3, axis=1)
        X = X.cpu().data.numpy()
        X = X * -(protected - 1)    # zero out the important region
        X = X + (protected * 0.5) # set x to 0.5 values
        X = Variable(torch.from_numpy(X).cuda(), requires_grad = True).float()        
        return X

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


def get_input_grad(x, output, y, create_graph=False):
        '''two methods for getting input gradient'''
        cross_entropy = True
        if cross_entropy:
            loss = F.cross_entropy(output, y)
            x_grad, = torch.autograd.grad(loss, x, create_graph=create_graph)
        else:
            grad_out = torch.zeros_like(output.data)
            grad_out.scatter_(1, y.data.unsqueeze(0).t(), 1.0)
            x_grad, = torch.autograd.grad(output, x,
                                          grad_outputs=grad_out,
                                          create_graph=create_graph)
        return x_grad

# uses no l1 (we want the adversary to attack everything. Though it could be extended to do a form of sparse adversarial examples[ref])
# uses a fixed L2, though the L2 could be set example specific using the change in label if we assume access to the model
# clips delta to be in the range (-e,e) rather than unbounded.
# uses a single-step attack (though can be easily extended to iterative)

# notice how the adversary is stronger. We leave future work to explore this in more detail.
def caso_perturb(model, x, y=None, epsilon=2.0/255.0, protected=None):
        # lambda_t1 = 1
        # lambda_t2 = 1
        # lambda_l2 = 100
        # n_iter = 10        

        # batch_size, n_chs, height, width = x.shape
        # delta = torch.zeros((batch_size, n_chs, height * width)).cuda()        
        # delta = nn.Parameter(delta, requires_grad=True)
        # optimizer = torch.optim.Adam([delta], lr=1e-4)
            
        # for i in range(self.n_iter):
        #     output = model(x)
        #     y = output.max(1)[1]

        #     x_grad = self.get_input_grad(x, output, y, create_graph=True)
        #     x_grad = x_grad.view((batch_size, n_chs, -1))

        #     hessian_delta_vp, = torch.autograd.grad(
        #         x_grad.dot(delta).sum(), x, create_graph=True)
        #     hessian_delta_vp = hessian_delta_vp.view((batch_size, n_chs, -1))
        #     taylor_1 = x_grad.dot(delta).sum()
        #     taylor_2 = 0.5 * delta.dot(hessian_delta_vp).sum()
        #     l2_term = F.mse_loss(delta, torch.zeros_like(delta))

        #     loss = (
        #         - self.lambda_t1 * taylor_1
        #         - self.lambda_t2 * taylor_2                
        #         + self.lambda_l2 * l2_term
        #     )
            
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
    
        # delta = delta.view((batch_size, n_chs, height, width)).data
               # lambda_t1=1,
               #   lambda_t2=1,
               #   lambda_l1=1e4,
               #   lambda_l2=1e5,
               #   n_iter=10,
               #   # optim='sgd',
               #   # lr=0.1,
               #   optim='adam',
               #   lr=1e-4,
               #   init='zero',
               #   times_input=False, 
        local_explainer = SparseExplainer(lambda_l1=0, lambda_l2=100)#, lambda_t2=0,lambda_l2=0)
        delta = local_explainer.explain(model, copy.deepcopy(x).data)        
        protected = np.repeat(protected[:, np.newaxis, :, :], 3, axis=1)
        adv_sign = delta.sign().cpu().numpy() * protected
        perturbed_X = x.data.cpu().numpy() + epsilon * adv_sign
        perturbed_X = np.clip(perturbed_X, 0, 1)
        return Variable(torch.from_numpy(perturbed_X).cuda(), requires_grad = True).float()
        
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
    #transf = get_preprocess('resnet50', 'sparse',dataset)
    #model = utils.load_model('cifar50')
    model = utils.load_model('resnet18')
    model.cuda()
    model.eval()

    explainers = [
        ('Vanilla', VanillaGradExplainer()),
        #('Random', None),
        #('SmoothGrad', SmoothGradExplainer()),        
        #('Tuned_Sparse', LambdaTunerExplainer()),                                        
        #('IntegratedGrad', IntegrateGradExplainer()),
    ]


    cutoff_scores = dict()
    for method_name, explainer in explainers:
        cutoff_scores[method_name] = [0] * 11

    cutoffs = [0,10,20,30,40,50,60,70,80,90,100]
    #cutoffs = [0,1,2,3,4,5,6,7,8,9,10]
    num_images = 16#4#128

    batch_size = 1#16
    attack_method = 'single_step'#'caso_perturb'
    batches = utils.load_data(batch_size=batch_size, num_images = num_images, transf=transf, dataset=dataset)
    for batch in batches:
        inputs = Variable(batch.cuda(), requires_grad=True)
        forward_pass = model(inputs)
        original_prediction = forward_pass.max(1, keepdim=True)[1]
        original_confidence = F.softmax(forward_pass, dim=1)
        #confidence_for_prediction = original_confidence[original_prediction]#.max(1, keepdim=True)
        confidence_for_class = original_confidence.cpu().data.numpy()[0][original_prediction.cpu().data.numpy()][0][0]

        #raw_img = batch.cpu().numpy()[0]#viz.pil_loader(batch.cpu().numpy()[0])
        #all_saliency_maps = []                
        # raw_img = batch.cpu().numpy()[0]#viz.pil_loader(batch.cpu().numpy()[0])
        #all_saliency_maps = []                

        for method_name, explainer in explainers:
            if method_name == "Random":
                saliency = torch.from_numpy(np.random.rand(*inputs.shape)).float()
            else:
                saliency = explainer.explain(model, copy.deepcopy(inputs).data)
            saliency = viz.VisualizeImageGrayscale(saliency.cpu())

            for cutoff in cutoffs:
                protected_region = attackImportant(saliency.cpu().numpy(), cutoff=cutoff)
                if attack_method == 'single_step':
                    adversarial_image = single_step_perturb(model, copy.deepcopy(inputs), protected = protected_region)                                          
                    adversarial_prediction = model(adversarial_image).max(1, keepdim=True)[1]
                    correct = original_prediction.eq(adversarial_prediction).sum().cpu().data.numpy()
                elif attack_method == 'caso_perturb':
                    adversarial_image = caso_perturb(model, copy.deepcopy(inputs), protected = protected_region)                                          
                    adversarial_prediction = model(adversarial_image).max(1, keepdim=True)[1]
                    correct = original_prediction.eq(adversarial_prediction).sum().cpu().data.numpy()
                elif attack_method == 'gray_out':
                    gray_image = gray_out(model, copy.deepcopy(inputs), protected = protected_region)
                    gray_confidence = F.softmax(model(gray_image), dim=1)
                    gray_confidence_for_class = gray_confidence.cpu().data.numpy()[0][original_prediction.cpu().data.numpy()][0][0]
                    correct = confidence_for_class - gray_confidence_for_class
                elif attack_method == "iterative":
                    adversarial_image = iterative_perturb(model, copy.deepcopy(inputs), protected = protected_region)
                    adversarial_prediction = model(adversarial_image).max(1, keepdim=True)[1]
                    correct = original_prediction.eq(adversarial_prediction).sum().cpu().data.numpy()
                elif attack_method == "second_order":
                    adversarial_image = vat_attack(model, copy.deepcopy(inputs), protected = protected_region)
                    adversarial_prediction = model(adversarial_image).max(1, keepdim=True)[1]
                    correct = original_prediction.eq(adversarial_prediction).sum().cpu().data.numpy()

                cutoff_scores[method_name][int(cutoff/10)] += float(correct / batch_size)
                            
                protected_region = np.repeat(protected_region[:, np.newaxis, :, :], 3, axis=1)                
                all_saliency_maps.append(batch.cpu().numpy()[0] * (1 - protected_region))# + (128 * protected_region))

        continue
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


    with open("protected_results.txt", "a") as text_file:
        text_file.write(str(method_name) + '\n')
        for cutoff in cutoffs:            
            text_file.write('\n' + str(cutoff) + '\n' +'\n')
            print("Adversary Can Modify: ", cutoff)
            for method_name, explainer in explainers:
                print(method_name, cutoff_scores[method_name][int(cutoff/10)] / (num_images / batch_size))
                text_file.write(str(method_name) + '\n')
                text_file.write(str(cutoff_scores[method_name][int(cutoff/10)] / (num_images / batch_size)) + '\n')                
                    


