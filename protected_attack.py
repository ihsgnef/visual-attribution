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

batch_size = 32

def perturb(model, X, y=None, epsilon=2.0/255.0, protected=None):
    output = model(X)
    if y is None:
        y = output.max(1)[1]
    loss = F.cross_entropy(output, y)
    loss.backward()
    grad_sign = X.grad.data.cpu().sign().numpy()
    #################################################################### TODO, use different masks?
    protected = np.repeat(protected[:, np.newaxis, :, :], 3, axis=1)
    #print(protected[0][0][0][0:100])
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
        #print(len(bottom_k[:k]))
        #np.put(sal, sal[bottom_k[:k]], 1)
        sal = np.reshape(sal, (height,width))
        new_saliency.append(sal)
    new_saliency = np.stack(new_saliency)
    assert new_saliency.shape == (batch_size, height, width)
    return new_saliency

def run_protected(inputs, cutoff):
    sparse_args = {
        'lambda_t1': 1,
        'lambda_t2': 1,
        'lambda_l1': 1e2,
        'lambda_l2': 1e4,
        'n_iterations': 10,
        'optim': 'sgd',
        'lr': 0.1,
    }

    configs = [
        ['sparse', sparse_args],
        ['vanilla_grad', None],
        ['robust_sparse', None],
        #['grad_x_input', None],
        #['smooth_grad', None],
        #['integrate_grad', None],
    ]

    inputs = Variable(inputs.cuda(), requires_grad=True)
    scores = dict()
    for method_name, kwargs in configs:
        if method_name == "random":
            saliency = torch.from_numpy(np.random.rand(len(inputs),3,32,32)).cuda()
            saliency = viz.VisualizeImageGrayscale(saliency.cpu())
            protected_region = attackUnImportant(saliency.cpu().numpy(), cutoff=cutoff)
        else:
            explainer = get_explainer(model, method_name, kwargs)
            saliency = explainer.explain(copy.deepcopy(inputs), None)
            saliency = viz.VisualizeImageGrayscale(saliency.cpu())
            protected_region = attackUnImportant(saliency.cpu().numpy(), cutoff=cutoff)

        adversarial_image = perturb(model, copy.deepcopy(inputs), protected = protected_region)
        original_prediction = model(inputs).max(1, keepdim=True)[1]
        adversarial_prediction = model(adversarial_image).max(1, keepdim=True)[1]
        correct = original_prediction.eq(adversarial_prediction).sum().cpu().data.numpy()
        scores[method_name] = float(correct / batch_size)
    return scores

if __name__ == '__main__':
    dataset = 'cifar10'
    transf = get_preprocess('resnet50', 'sparse',dataset)
    model = utils.load_model('cifar50')
    model.cuda()
    model.eval()

    cutoffs = [60,70]#0,10,20,30,40,50,60,70,80,90] # percentage adversary can see
    num_images = 32
    for cutoff in cutoffs:
        batches = utils.load_data(batch_size=batch_size, num_images = num_images, transf=transf, dataset=dataset)
        all_scores = None
        for batch in batches:
            scores = run_protected(batch, cutoff)
            if all_scores is None:
                all_scores = scores
                continue
            for method_name in scores:
                all_scores[method_name] += scores[method_name]

        with open("protected_results.txt", "a") as text_file:
            print("Adversary Can Modify: ", cutoff)
            text_file.write('\n' + str(cutoff) + '\n' +'\n')
            for method_name in all_scores:
                accuracy = all_scores[method_name] / float(num_images / batch_size)
                print(method_name, accuracy)
                text_file.write(str(method_name) + '\n')
                text_file.write(str(accuracy) + '\n')
