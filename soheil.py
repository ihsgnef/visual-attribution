import sys
sys.path.append('./')

from create_explainer import get_explainer
from preprocess import get_preprocess
import utils
import viz
import torch
import time
import os
import pylab
import numpy as np
import torch.nn.functional as F

params = {
    'font.family': 'sans-serif',
    'axes.titlesize': 25,
    'axes.titlepad': 10,
}
pylab.rcParams.update(params)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class SoheilExplainer(object):
    def __init__(self, model):
        self.model = model        
        self.lambda_1 = 0.1
        self.lambda_2 = 0.1
        self.n_iterations = 2
    
    def explain(self, inp, ind=None):
        delta = torch.nn.Parameter(torch.zeros_like(inp.data).cuda(), requires_grad=True)
        optimizer = torch.optim.SGD([delta], lr=0.1)
        for i in range(self.n_iterations):
            output = self.model(inp)
            if ind is None:
                ind = output.max(1)[1]
            grad_out = F.cross_entropy(output, ind)
            inp_grad, = torch.autograd.grad(grad_out, inp, create_graph=True)
            hessian_delta_vp, = torch.autograd.grad((inp_grad @ delta).sum(), inp, create_graph=True)
            first_order = inp_grad.dot(delta).sum()
            second_order = 0.5 * (delta.t() @ hessian_delta_vp).sum()
            l1_term = F.l1_loss(delta, torch.zeros_like(delta))
            l2_term = F.mse_loss(delta, torch.zeros_like(delta))
            print(first_order.data, second_order.data, l1_term.data, l2_term.data)
            loss = - first_order - second_order + self.lambda_1 * l1_term + self.lambda_2 * l2_term
            # delta_grad, = torch.autograd.grad(loss, delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return delta

image_path = 'images/elephant.png'
image_class = 101 # tusker
raw_img = viz.pil_loader(image_path)
        
transf = get_preprocess('resnet50', 'vanilla_grad')
model = utils.load_model('resnet50')
model.cuda()
explainer = SoheilExplainer(model)
    
inp = transf(raw_img)
inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)

target = torch.LongTensor([image_class]).cuda()
saliency = explainer.explain(inp)
# saliency = utils.upsample(saliency, (raw_img.height, raw_img.width))
