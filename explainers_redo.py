import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import viz
import math

def zero_grad(x):
    if isinstance(x, Variable):
        if x.grad is not None:
            x.grad.data.zero_()
    elif isinstance(x, torch.nn.Module):
        for p in x.parameters():
                if p.grad is not None:
                    p.grad.data.zero_()


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, 2, dim=1, keepdim=True) + 1e-8
    return d


class SparseExplainer:

    def __init__(self,
                 lambda_t1=1,
                 lambda_t2=1,
                 lambda_l1=1e2,
                 lambda_l2=0,
                 n_iter=10,
                 # optim='sgd',
                 # lr=0.1,
                 optim='adam',
                 lr=1e-4,
                 init='zero',
                 times_input=False,
                 ):
        '''
        :param lambda_t1
        :param lambda_t2
        :param lambda_l1
        :param lambda_l2
        :param n_iter
        :param optim
        :param lr
        :param init: [zero, random, grad]  # TODO add tensor init
        :param times_input: multiple with input at the end
        '''
        self.lambda_t1 = lambda_t1
        self.lambda_t2 = lambda_t2
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.n_iter = int(n_iter)
        self.optim = optim.lower()
        self.lr = lr
        self.init = init
        self.times_input = times_input
        assert init in ['zero', 'random', 'grad']
        self.history = defaultdict(list)

    def get_input_grad(self, x, output, y, create_graph=False):
        '''two methods for getting input gradient'''
        loss = F.cross_entropy(output, y)
        x_grad, = torch.autograd.grad(loss, x, create_graph=create_graph)

        # grad_out = torch.zeros_like(output.data)
        # grad_out.scatter_(1, y.data.unsqueeze(0).t(), 1.0)
        # x_grad, = torch.autograd.grad(output, x,
        #                               grad_outputs=grad_out,
        #                               create_graph=create_graph)
        return x_grad

    def initialize_delta(self, model, x):
        batch_size, n_chs, height, width = x.shape
        if self.init == 'zero':
            delta = torch.zeros((batch_size, n_chs, height * width)).cuda()
        elif self.init == 'grad':
            output = model(x)
            y = output.max(1)[1]
            delta = self.get_input_grad(x, output, y).data
            delta = delta.view(batch_size, n_chs, -1)
        elif self.init == 'random':
            delta = torch.rand((batch_size, n_chs, height * width))
            delta = delta.sub(0.5).cuda()
            delta = _l2_normalize(delta)
        delta = nn.Parameter(delta, requires_grad=True)
        return delta

    def explain(self, model, x):
        self.history = defaultdict(list)

        x = Variable(x, requires_grad=True)
        delta = self.initialize_delta(model, x)
        batch_size, n_chs, height, width = x.shape

        if self.optim == 'sgd':
            optimizer = torch.optim.SGD([delta], lr=self.lr, momentum=0.9)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam([delta], lr=self.lr)

        for i in range(self.n_iter):
            output = model(x)
            y = output.max(1)[1]

            x_grad = self.get_input_grad(x, output, y, create_graph=True)
            x_grad = x_grad.view((batch_size, n_chs, -1))

            hessian_delta_vp, = torch.autograd.grad(
                x_grad.dot(delta).sum(), x, create_graph=True)
            hessian_delta_vp = hessian_delta_vp.view((batch_size, n_chs, -1))
            taylor_1 = x_grad.dot(delta).sum()
            taylor_2 = 0.5 * delta.dot(hessian_delta_vp).sum()
            l1_term = F.l1_loss(delta, torch.zeros_like(delta))
            l2_term = F.mse_loss(delta, torch.zeros_like(delta))

            loss = (
                - self.lambda_t1 * taylor_1
                - self.lambda_t2 * taylor_2
                + self.lambda_l1 * l1_term
                + self.lambda_l2 * l2_term
            )

            # log optimization
            vmax = delta.abs().sum(1).max(1)[0]
            vmin = delta.abs().sum(1).min(1)[0]
            self.history['l1'].append(l1_term.data.cpu().numpy())
            self.history['l2'].append(l2_term.data.cpu().numpy())
            self.history['grad'].append(taylor_1.data.cpu().numpy())
            self.history['hessian'].append(taylor_2.data.cpu().numpy())
            self.history['vmax'].append(vmax.data.cpu().numpy())
            self.history['vmin'].append(vmin.data.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        delta = delta.view((batch_size, n_chs, height, width)).data
        if self.times_input:
            delta *= x.data
        return delta


class RobustSparseExplainer(SparseExplainer):

    def __init__(self,
                 lambda_t1=1,
                 lambda_t2=1,
                 lambda_l1=1e4,
                 lambda_l2=1e5,
                 n_iter=10,
                 # optim='sgd',
                 # lr=0.1,
                 optim='adam',
                 lr=1e-4,
                 init='zero',
                 times_input=False,
                 ):
        super(RobustSparseExplainer, self).__init__(lambda_t1, lambda_t2,
                                                    lambda_l1, lambda_l2,
                                                    n_iter, optim, lr, init,
                                                    times_input)

    def explain(self, model, x):
        self.history = defaultdict(list)

        x = Variable(x, requires_grad=True)
        delta = self.initialize_delta(model, x)
        batch_size, n_chs, height, width = x.shape

        if self.optim == 'sgd':
            optimizer = torch.optim.SGD([delta], lr=self.lr, momentum=0.9)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam([delta], lr=self.lr)

        # + g^t Delta
        # - \lambda_t2 |(\Delta-g/2)^t H g|
        # - \lambda_l2 |\Delta|_2^2
        # - \lambda_l1 |\Delta|_1

        for i in range(self.n_iter):
            output = model(x)
            y = output.max(1)[1]

            x_grad = self.get_input_grad(x, output, y, create_graph=True)
            x_grad = x_grad.view((batch_size, n_chs, -1))

            g = x_grad.clone()
            g.detach()

            hessian_delta_vp, = torch.autograd.grad(
                x_grad.dot(g).sum(), x, create_graph=True)
            hessian_delta_vp = hessian_delta_vp.view((batch_size, n_chs, -1))
            taylor_2 = (delta - g / 2).dot(hessian_delta_vp)
            taylor_2 = F.l1_loss(taylor_2, torch.zeros_like(taylor_2))
            taylor_1 = x_grad.dot(delta).sum()
            l1_term = F.l1_loss(delta, torch.zeros_like(delta))
            l2_term = F.mse_loss(delta, torch.zeros_like(delta))

            # log optimization
            vmax = delta.abs().sum(1).max(1)[0]
            vmin = delta.abs().sum(1).min(1)[0]
            self.history['l1'].append(l1_term.data.cpu().numpy())
            self.history['l2'].append(l2_term.data.cpu().numpy())
            self.history['grad'].append(taylor_1.data.cpu().numpy())
            self.history['hessian'].append(taylor_2.data.cpu().numpy())
            self.history['vmax'].append(vmax.data.cpu().numpy())
            self.history['vmin'].append(vmin.data.cpu().numpy())

            loss = (
                - self.lambda_t1 * taylor_1
                + self.lambda_t2 * taylor_2
                + self.lambda_l1 * l1_term
                + self.lambda_l2 * l2_term
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        delta = delta.view((batch_size, n_chs, height, width)).data
        if self.times_input:
            delta *= x.data
        return delta

# compute median difference
def get_median_difference(saliency):
    saliency = viz.VisualizeImageGrayscale(saliency.cpu())    
    s = saliency.cpu().numpy().ravel()
    topk = np.argsort(-s)[:int(s.size * 0.02)]
    botk = np.argsort(s)[:int(s.size * 0.98)]
    top_median = np.median(s[topk])
    bot_median = np.median(s[botk])        
    return top_median - bot_median

class LambdaTunerExplainer:
    def explain(self, model, x):                
        best_median = 0                
        current_lambda1 = 0
        current_lambda2 = 0

        # get initial explanation
        saliency = SparseExplainer(lambda_l1=current_lambda1, lambda_l2=current_lambda2).explain(model, x)
        current_median_difference = get_median_difference(saliency)
        print('lambda_1', current_lambda1, 'lambda_2', current_lambda2, 'current_median_difference', current_median_difference)        

        current_lambda1 = 0.01 # Need to start at non-zero because 10*0 = 0
        # also note these values are multiplied by 10 immediately
        current_lambda2 = 1#10
        increase_rate = 2#10  # multiply by 10 each time

        # lambda_l1 search
        while (current_median_difference >= best_median or abs(current_median_difference - best_median) < .01):
            best_median = current_median_difference # update best median
            current_lambda1 = current_lambda1 * increase_rate
            
            saliency = SparseExplainer(lambda_l1=current_lambda1, lambda_l2=current_lambda2).explain(model, x)
            current_median_difference = get_median_difference(saliency)
            print('lambda_1', current_lambda1, 'lambda_2', current_lambda2, 'current_median_difference', current_median_difference)
        
        print("Done Tuning Lambda_L1")
        current_lambda1 = current_lambda1 / increase_rate # because current settings are one too far here        
        current_median_difference = best_median

        while (current_median_difference >= best_median or abs(current_median_difference - best_median) < .01):
            best_median = current_median_difference # update best median
            current_lambda2 = current_lambda2 * increase_rate

            saliency = SparseExplainer(lambda_l1=current_lambda1, lambda_l2=current_lambda2).explain(model, x)
            current_median_difference = get_median_difference(saliency)
            print('lambda_1', current_lambda1, 'lambda_2', current_lambda2, 'current_median_difference', current_median_difference)

        print("Done Tuning Lambda_L2")            
        current_lambda2 = current_lambda2 / increase_rate # because current settings are one too far here        

        saliency = SparseExplainer(lambda_l1=current_lambda1, lambda_l2=current_lambda2).explain(model, x)
        current_median_difference = get_median_difference(saliency)
        print('Final Lambda_1', current_lambda1, 'Final_Lambda_2', current_lambda2, 'Final_Median', current_median_difference)
        return saliency


class VanillaGradExplainer:

    def __init__(self, times_input=False):
        self.times_input = times_input

    def get_input_grad(self, x, output, y, create_graph=False):
        '''two methods for getting input gradient'''
        loss = F.cross_entropy(output, y)
        x_grad, = torch.autograd.grad(loss, x, create_graph=create_graph)

        # grad_out = torch.zeros_like(o.utput.data)
        # grad_out.scatter_(1, y.data.unsqueeze(0).t(), 1.0)
        # x_grad, = torch.autograd.grad(output, x,
        #                               grad_outputs=grad_out,
        #                               create_graph=create_graph)
        return x_grad

    def explain(self, model, x):
        x = Variable(x, requires_grad=True)
        output = model(x)
        y = output.max(1)[1]
        x_grad = self.get_input_grad(x, output, y).data
        if self.times_input:
            x_grad *= x.data
        return x_grad


class IntegrateGradExplainer(VanillaGradExplainer):
    def __init__(self, n_iter=100, times_input=False):
        self.n_iter = n_iter
        self.times_input = times_input

    def explain(self, model, x):
        grad = 0
        x_data = x.clone()
        for alpha in np.arange(1 / self.n_iter, 1.0, 1 / self.n_iter):
            x_var = Variable(x_data * alpha, requires_grad=True)
            output = model(x_var)
            y = output.max(1)[1]
            g = self.get_input_grad(x_var, output, y)
            grad += g.data
        if self.times_input:
            grad *= x_data
        return grad / self.n_iter


class SmoothGradExplainer(object):
    def __init__(self, base_explainer=None, stdev_spread=0.15,
                 nsamples=25, magnitude=True, times_input=False):
        if base_explainer is None:
            base_explainer = VanillaGradExplainer()
        self.base_explainer = base_explainer
        self.stdev_spread = stdev_spread
        self.nsamples = nsamples
        self.magnitude = magnitude
        self.times_input = times_input

    def explain(self, model, x):
        stdev = self.stdev_spread * (x.max() - x.min())
        total_gradients = 0
        for i in range(self.nsamples):
            noise = torch.randn(x.shape).cuda() * stdev
            x_var = noise + x.clone()
            grad = self.base_explainer.explain(model, x_var)
            if self.magnitude:
                total_gradients += grad ** 2
            else:
                total_gradients += grad
        total_gradients /= self.nsamples
        if self.times_input:
            total_gradients *= x
        return total_gradients

