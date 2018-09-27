import numpy as np
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import viz


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


def _kl_div(log_probs, probs):
    # pytorch KLDLoss is averaged over all dim if size_average=True
    kld = F.kl_div(log_probs, probs, size_average=False)
    return kld / log_probs.shape[0]


class VanillaGradExplainer:

    def __init__(self, times_input=False):
        self.times_input = times_input

    def get_input_grad(self, x, output, y, create_graph=False):
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

    def explain(self, model, x):
        x = Variable(x, requires_grad=True)
        output = model(x)
        y = output.max(1)[1]
        x_grad = self.get_input_grad(x, output, y).data
        if self.times_input:
            x_grad *= x.data
        return x_grad


class VATExplainer:

    def __init__(self, xi=1e-6, n_iterations=1, times_input=False):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param n_iterations: number of iterations (default: 1)
        """
        self.xi = xi
        self.times_input = times_input

    def explain(self, model, x, KL=True):
        x_var = Variable(x.clone(), requires_grad=True)
        output = model(x_var)
        y = output.max(1)[1]
        if KL:
            pred = F.log_softmax(model(x_var), dim=1).detach()
        d = torch.rand(x_var.shape).sub(0.5).cuda()
        d = _l2_normalize(d)
        for _ in range(self.n_iterations):
            model.zero_grad()
            d = Variable(self.xi * d, requires_grad=True)
            pred_hat = model(x_var + d)
            if KL:
                adv_loss = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
            else:
                adv_loss = F.cross_entropy(pred_hat, y)
            d_grad, = torch.autograd.grad(adv_loss, d)
            d = _l2_normalize(d_grad.data)
        if self.times_input:
            d *= x
        return d


class Eigenvalue(VanillaGradExplainer):

    def explain(self, model, x):
        batch_size, n_chs, height, width = x.shape
        x_data = x.clone()
        x = Variable(x, requires_grad=True)
        d = torch.rand(batch_size, n_chs, height * width)
        d = _l2_normalize(d.sub(0.5)).cuda()

        for _ in range(1):
            model.zero_grad()
            output = model(x)
            y = output.max(1)[1]
            loss = F.cross_entropy(output, y)
            x_grad, = torch.autograd.grad(loss, x, create_graph=True)
            x_grad = x_grad.view(batch_size, n_chs, -1)
            d = Variable(d, requires_grad=True)
            hvp, = torch.autograd.grad(x_grad.dot(d).sum(), x)
            hvp = hvp.data.view(batch_size, n_chs, -1)
            taylor_2 = (d * hvp).sum()
            d = _l2_normalize(hvp).view(batch_size, n_chs, -1)
            print(taylor_2)
        return VanillaGradExplainer().explain(model, x_data)


class SparseExplainer(VanillaGradExplainer):

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
        assert init in ['zero', 'random', 'grad', 'vat']
        self.history = defaultdict(list)

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
        elif self.init == 'vat':
            delta = VATExplainer().explain(model, x.data)
            delta = delta.view(batch_size, n_chs, height * width)
        delta = nn.Parameter(delta, requires_grad=True)
        return delta

    def loss(self, taylor_1, taylor_2, l1_term, l2_term):
        batch_size = l1_term.shape[0]
        taylor_2 = (self.lambda_t2 * taylor_2).sum() / batch_size
        l1_term = (self.lambda_l1 * l1_term).sum() / batch_size
        l2_term = (self.lambda_l2 * l2_term).sum() / batch_size
        loss = (
            - self.lambda_t1 * taylor_1
            - taylor_2
            + l1_term
            + l2_term
        )
        return loss

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
            taylor_2 = 0.5 * (delta * hessian_delta_vp)
            taylor_2 = taylor_2.sum(2).sum(1) / (n_chs * height * width)
            # taylor_2 = 0.5 * delta.dot(hessian_delta_vp).sum()
            # l1_term = F.l1_loss(delta, torch.zeros_like(delta))
            # l2_term = F.mse_loss(delta, torch.zeros_like(delta))
            l1_term = F.l1_loss(delta, torch.zeros_like(delta), reduce=False)
            l2_term = F.mse_loss(delta, torch.zeros_like(delta), reduce=False)
            l1_term = l1_term.sum(2).sum(1) / (n_chs * height * width)
            l2_term = l2_term.sum(2).sum(1) / (n_chs * height * width)
            # leave batch unreduced so each example in the batch can use
            # different hyperparameters

            # loss = (
            #     - self.lambda_t1 * taylor_1
            #     - self.lambda_t2 * taylor_2
            #     + self.lambda_l1 * l1_term
            #     + self.lambda_l2 * l2_term
            # )

            loss = self.loss(taylor_1, taylor_2, l1_term, l2_term)

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
                 optim='adam',
                 lr=1e-4,
                 init='zero',
                 times_input=False,
                 ):
        super(RobustSparseExplainer, self).__init__(lambda_t1, lambda_t2,
                                                    lambda_l1, lambda_l2,
                                                    n_iter, optim, lr, init,
                                                    times_input)

    def loss(self, taylor_1, taylor_2, l1_term, l2_term):
        batch_size = l1_term.shape[0]
        taylor_2 = (self.lambda_t2 * taylor_2).sum() / batch_size
        l1_term = (self.lambda_l1 * l1_term).sum() / batch_size
        l2_term = (self.lambda_l2 * l2_term).sum() / batch_size
        loss = (
            - self.lambda_t1 * taylor_1
            + taylor_2
            + l1_term
            + l2_term
        )
        return loss

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

            taylor_1 = x_grad.dot(delta).sum()
            # taylor_2 = (delta - g / 2).dot(hessian_delta_vp)
            taylor_2 = (delta - g / 2) * hessian_delta_vp
            taylor_2 = F.l1_loss(taylor_2, torch.zeros_like(taylor_2),
                                 reduce=False)
            taylor_2 = taylor_2.sum(2).sum(1) / (n_chs * height * width)

            l1_term = F.l1_loss(delta, torch.zeros_like(delta), reduce=False)
            l2_term = F.mse_loss(delta, torch.zeros_like(delta), reduce=False)
            l1_term = l1_term.sum(2).sum(1) / (n_chs * height * width)
            l2_term = l2_term.sum(2).sum(1) / (n_chs * height * width)

            # loss = (
            #     - self.lambda_t1 * taylor_1
            #     + self.lambda_t2 * taylor_2
            #     + self.lambda_l1 * l1_term
            #     + self.lambda_l2 * l2_term
            # )

            loss = self.loss(taylor_1, taylor_2, l1_term, l2_term)

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


class LambdaTunerExplainer:

    def __init__(self, times_input=False):
        self.times_input = times_input

    def explain(self, model, x, get_lambdas=False, ):
        input_data = x.clone()

        best_median = 0
        lambda_1 = 0
        lambda_2 = 0

        # get initial explanation
        saliency = SparseExplainer(lambda_l1=lambda_1,
                                   lambda_l2=lambda_2).explain(model, x)
        current_median_difference = viz.get_median_difference(
            viz.agg_clip(saliency.cpu().numpy()))
        print('lambda_1', lambda_1, 'lambda_2', lambda_2,
              'current_median_difference', current_median_difference)

        lambda_1 = 0.50  # Need to start at non-zero because 10*0 = 0
        # also note these values are multiplied by 10 immediately
        lambda_2 = 100
        increase_rate = 2  # multiply each time
        patience = 0.02

        # lambda_l1 search
        while (current_median_difference >= best_median
               or abs(current_median_difference - best_median) < patience):
            best_median = max(current_median_difference, best_median)
            if best_median > 0.945:  # return
                saliency = SparseExplainer(
                    lambda_l1=lambda_1, lambda_l2=lambda_2).explain(model, x)
                current_median_difference = viz.get_median_difference(
                    viz.agg_clip(saliency.cpu().numpy()))
                print('Final Lambda_1', lambda_1,
                      'Final_Lambda_2', lambda_2,
                      'Final_Median', current_median_difference)

                if self.times_input:
                    saliency *= input_data

                if get_lambdas:
                    return saliency, lambda_1, lambda_2
                else:
                    return saliency

            lambda_1 = lambda_1 * increase_rate

            saliency = SparseExplainer(
                lambda_l1=lambda_1,
                lambda_l2=lambda_2).explain(model, x)
            current_median_difference = viz.get_median_difference(
                viz.agg_clip(saliency.cpu().numpy()))
            print('lambda_1', lambda_1, 'lambda_2', lambda_2,
                  'current_median_difference', current_median_difference)

        print("Done Tuning Lambda_L1")
        # because current settings are one too far here
        lambda_1 = lambda_1 / increase_rate
        current_median_difference = best_median

        while (current_median_difference >= best_median
               or abs(current_median_difference - best_median) < patience):
            best_median = max(current_median_difference, best_median)
            if best_median > 0.945:  # return
                saliency = SparseExplainer(
                    lambda_l1=lambda_1, lambda_l2=lambda_2).explain(model, x)
                current_median_difference = viz.get_median_difference(
                    viz.agg_clip(saliency.cpu().numpy()))
                print('Final Lambda_1', lambda_1,
                      'Final_Lambda_2', lambda_2,
                      'Final_Median', current_median_difference)

                if self.times_input:
                    saliency *= input_data

                if get_lambdas:
                    return saliency, lambda_1, lambda_2
                else:
                    return saliency

            lambda_2 = lambda_2 * increase_rate

            saliency = SparseExplainer(
                lambda_l1=lambda_1,
                lambda_l2=lambda_2).explain(model, x)
            current_median_difference = viz.get_median_difference(
                viz.agg_clip(saliency.cpu().numpy()))
            print('lambda_1', lambda_1, 'lambda_2', lambda_2,
                  'current_median_difference', current_median_difference)

        print("Done Tuning Lambda_L2")
        # because current settings are one too far here
        lambda_2 = lambda_2 / increase_rate

        saliency = SparseExplainer(
            lambda_l1=lambda_1, lambda_l2=lambda_2).explain(model, x)
        current_median_difference = viz.get_median_difference(
            viz.agg_clip(saliency.cpu().numpy()))
        print('Final Lambda_1', lambda_1,
              'Final_Lambda_2', lambda_2,
              'Final_Median', current_median_difference)

        if self.times_input:
            saliency *= input_data

        if get_lambdas:
            return saliency, lambda_1, lambda_2
        else:
            return saliency


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
        grad = grad / self.n_iter
        return grad


class SmoothGradExplainer:

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


class SmoothCASO:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def explain(self, model, x):
        tuner = BatchTuner(SparseExplainer, **self.kwargs)
        saliency, lambdas = tuner.explain(model, x, get_lambdas=True)
        lambda_vectors = {k: [] for k in lambdas[0].keys()}
        for key in lambda_vectors:
            ls = [l[key] for l in lambdas]
            lambda_vectors[key] = Variable(torch.FloatTensor(ls).cuda())
        exp = SmoothGradExplainer(SparseExplainer(**lambda_vectors),
                                  nsamples=25, magnitude=False)
        return exp.explain(model, x)


class BatchTuner:

    def __init__(self, Exp,
                 l1_lo=1e-1, l1_hi=2e5,
                 l2_lo=1, l2_hi=1e6,
                 t1_lo=1, t1_hi=1,
                 t2_lo=1, t2_hi=1,
                 n_steps=16, n_search=3,
                 lambda_t1=None, lambda_t2=None,
                 lambda_l1=None, lambda_l2=None,
                 n_iter=10, optim='adam', lr=1e-4, init='zero',
                 times_input=False):
        self.sparse_args = {
            'n_iter': n_iter,
            'optim': optim,
            'lr': lr,
            'init': init,
            'times_input': times_input,
        }
        self.Exp = Exp
        if lambda_t1 is not None:
            t1_lo = lambda_t1
            t1_hi = lambda_t1
        if lambda_t2 is not None:
            t2_lo = lambda_t2
            t2_hi = lambda_t2
        if lambda_l1 is not None:
            l1_lo = lambda_l1
            l1_hi = lambda_l1
        if lambda_l2 is not None:
            l2_lo = lambda_l2
            l2_hi = lambda_l2
        self.tunables = OrderedDict({
            'lambda_t1': (t1_lo, t1_hi),
            'lambda_t2': (t2_lo, t2_hi),
            'lambda_l1': (l1_lo, l1_hi),
            'lambda_l2': (l2_lo, l2_hi),
        })
        self.n_steps = n_steps
        self.n_search = n_search

    def explain_one(self, model, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        tunables = self.tunables.copy()
        best_lambdas = {key: lo for key, (lo, hi) in tunables.items()}
        best_median = 0
        best_saliency = 0
        # creat batch
        for i in range(self.n_search):
            xx = x.repeat(self.n_steps, 1, 1, 1)
            for param, (lo, hi) in tunables.items():
                if lo == hi:
                    continue
                ps = np.geomspace(lo, hi, self.n_steps)
                args = self.sparse_args.copy()
                args.update(best_lambdas)
                args[param] = Variable(torch.FloatTensor(ps).cuda())
                model.zero_grad()
                saliency = self.Exp(**args).explain(model, xx).cpu().numpy()
                s = viz.agg_clip(saliency)
                medians = [viz.get_median_difference(x) for x in s]
                best_idx = np.argmax(medians)
                best_lambdas[param] = ps[best_idx]
                lo = ps[max(best_idx - 1, 0)]
                hi = ps[min(best_idx + 1, len(ps) - 1)]
                tunables[param] = (lo, hi)
                if medians[best_idx] > best_median:
                    best_median = medians[best_idx]
                    best_saliency = saliency[best_idx]
                if best_median > 0.945:
                    break
        output = '{}: {:.3f}'.format(i, best_median)
        for param, best in best_lambdas.items():
            if isinstance(best, float):
                output += ' {}: {:.3f} '.format(param, best)
        print(output)
        return best_saliency, best_lambdas

    def explain(self, model, xs, get_lambdas=False):
        batch_size, n_chs, height, width = xs.shape
        saliency, lambdas = [], []
        for x in xs:
            s, ls = self.explain_one(model, x)
            saliency.append(s)
            lambdas.append(ls)
        saliency = torch.FloatTensor(np.stack(saliency)).cuda()
        if get_lambdas:
            return saliency, lambdas
        else:
            return saliency
