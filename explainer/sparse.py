import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict


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

    def __init__(self, model,
                 lambda_t1=1, lambda_t2=1,
                 lambda_l1=1e4, lambda_l2=1e4,
                 n_iterations=10, optim='sgd', lr=0.1,
                 times_input=False, init='zero'):
        self.model = model
        self.lambda_t1 = lambda_t1
        self.lambda_t2 = lambda_t2
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.n_iterations = int(n_iterations)
        self.optim = optim.lower()
        self.lr = lr
        self.times_input = times_input
        self.init = init
        assert init in ['zero', 'random', 'grad']
        self.history = defaultdict(list)

    def _backprop(self, inp, ind):
        zero_grad(self.model)
        output = self.model(inp)
        ind = output.data.max(1)[1]
        grad_out = torch.zeros_like(output.data)
        grad_out.scatter_(1, ind.unsqueeze(0).t(), 1.0)
        # output.backward(grad_out, create_graph=True)
        inp_grad, = torch.autograd.grad(output, inp, grad_outputs=grad_out,
                                        create_graph=True)
        return inp.grad

    def explain(self, inp, ind=None):
        self.history = defaultdict(list)

        batch_size, n_chs, height, width = inp.shape
        img_size = height * width
        if self.init == 'zero':
            delta = torch.zeros((batch_size, n_chs, img_size)).cuda()
        elif self.init == 'grad':
            output = self.model(inp)
            out_loss = F.cross_entropy(output, output.max(1)[1])
            delta = torch.autograd.grad(out_loss, inp)[0].data
            delta = delta.view(batch_size, n_chs, img_size)
        elif self.init == 'random':
            delta = torch.rand((batch_size, n_chs, img_size))
            delta = delta.sub(0.5).cuda()
            delta = _l2_normalize(delta)
        delta = nn.Parameter(delta, requires_grad=True)

        if self.optim == 'sgd':
            optimizer = torch.optim.SGD([delta], lr=self.lr, momentum=0.9)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam([delta], lr=self.lr)

        for i in range(self.n_iterations):
            output = self.model(inp)
            # if ind is None:
            ind = output.max(1)[1]
            # TODO, find a way to not recalculate grad each time

            '''two methods for getting input gradient'''
            # inp_grad = self._backprop(inp, ind)
            out_loss = F.cross_entropy(output, ind)
            inp_grad, = torch.autograd.grad(out_loss, inp, create_graph=True)

            inp_grad = inp_grad.view((batch_size, n_chs, img_size))

            hessian_delta_vp, = torch.autograd.grad(
                    inp_grad.dot(delta).sum(), inp, create_graph=True)
            hessian_delta_vp = hessian_delta_vp.view(
                    (batch_size, n_chs, img_size))
            taylor_1 = inp_grad.dot(delta).sum()
            taylor_2 = 0.5 * delta.dot(hessian_delta_vp).sum()
            l1_term = F.l1_loss(delta, torch.zeros_like(delta))
            l2_term = F.mse_loss(delta, torch.zeros_like(delta))

            loss = (
                - self.lambda_t1 * taylor_1
                - self.lambda_t2 * taylor_2
                + self.lambda_l1 * l1_term
                + self.lambda_l2 * l2_term
            )

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
            delta *= inp.data
        return delta


class RobustSparseExplainer:

    def __init__(self, model,
                 lambda_t1=1, lambda_t2=1,
                 lambda_l1=1e4, lambda_l2=1e4,
                 n_iterations=10, optim='sgd', lr=0.1,
                 times_input=False, init='zero'):
        self.model = model
        self.lambda_t1 = lambda_t1
        self.lambda_t2 = lambda_t2
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.n_iterations = int(n_iterations)
        self.optim = optim.lower()
        self.lr = lr
        self.times_input = times_input
        self.init = init
        assert init in ['zero', 'random', 'grad']
        self.history = defaultdict(list)

    def _backprop(self, inp, ind):
        zero_grad(self.model)
        output = self.model(inp)
        ind = output.data.max(1)[1]
        grad_out = torch.zeros_like(output.data)
        grad_out.scatter_(1, ind.unsqueeze(0).t(), 1.0)
        # output.backward(grad_out, create_graph=True)
        inp_grad, = torch.autograd.grad(output, inp, grad_outputs=grad_out,
                                        create_graph=True)
        return inp.grad

    def explain(self, inp, ind=None):
        self.history = defaultdict(list)
        batch_size, n_chs, height, width = inp.shape
        img_size = height * width
        '''initialize delta with zero or grad'''
        delta = torch.zeros((batch_size, n_chs, img_size)).cuda()
        # output = self.model(inp)
        # out_loss = F.cross_entropy(output, output.max(1)[1])
        # delta = torch.autograd.grad(out_loss, inp)[0].data
        # delta = delta.view(batch_size, n_chs, img_size)
        delta = nn.Parameter(delta, requires_grad=True)

        if self.optim == 'sgd':
            optimizer = torch.optim.SGD([delta], lr=self.lr, momentum=0.9)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam([delta], lr=self.lr)

        # + g^t Delta
        # - \lambda_t2 |(\Delta-g/2)^t H g|
        # - \lambda_l2 |\Delta|_2^2
        # - \lambda_l1 |\Delta|_1

        for i in range(self.n_iterations):
            output = self.model(inp)
            # if ind is None:
            ind = output.max(1)[1]

            '''two methods for getting input gradient'''
            # inp_grad = self._backprop(inp, ind)
            out_loss = F.cross_entropy(output, ind)
            inp_grad, = torch.autograd.grad(out_loss, inp, create_graph=True)

            inp_grad = inp_grad.view((batch_size, n_chs, img_size))
            g = inp_grad.clone()
            g.detach()

            hessian_delta_vp, = torch.autograd.grad(
                    inp_grad.dot(g).sum(), inp, create_graph=True)
            hessian_delta_vp = hessian_delta_vp.view(
                    (batch_size, n_chs, img_size))
            taylor_2 = (delta - g / 2).dot(hessian_delta_vp)
            taylor_2 = F.l1_loss(taylor_2, torch.zeros_like(taylor_2))
            taylor_1 = inp_grad.dot(delta).sum()
            l1_term = F.l1_loss(delta, torch.zeros_like(delta))
            l2_term = F.mse_loss(delta, torch.zeros_like(delta))

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
            delta *= inp.data
        return delta
