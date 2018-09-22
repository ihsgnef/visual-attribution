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

    def __init__(self,
                 lambda_t1=1,
                 lambda_t2=1,
                 lambda_l1=1e-1,
                 lambda_l2=1e4,
                 n_iter=10,
                 optim='sgd',
                 lr=0.1,
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
                 lambda_l1=1e-1,
                 lambda_l2=1e4,
                 n_iter=10,
                 optim='sgd',
                 lr=0.1,
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
