import numpy as np
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import viz


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, 2, dim=1, keepdim=True) + 1e-8
    return d


def _kl_div(log_probs, probs):
    kld = F.kl_div(log_probs, probs, size_average=False)
    return kld / log_probs.shape[0]


class Explainer:

    def get_input_grad(self, x, output, y, create_graph=False,
                       cross_entropy=True):
        '''Compute gradient of loss w.r.t input x.
        Args:
            x (torch.autograd.Variable):
                Input variable.
            output (torch.autograd.Variable):
                Output distribution after softmax.
            y (torch.autograd.Variable):
                Class label.
            create_graph:
                Set to True if higher-order gradient will be used.
            cross_entropy:
                Use by default the cross entropy loss.
        Rerturns:
            torch.autograd.Variable:
                The gradient, shape identical to x.
        '''
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

    def explain(self, model, x, y=None):
        '''Explain model prediction of y given x.
        Args:
            model (torch.nn.Module):
                The model to explain.
            x (torch.cuda.FloatTensor):
                Input tensor.
            y (torch.cuda.LongTensor or None):
                The class to explain. If None use prediction.
        Returns
            torch.cuda.FloatTensor:
                Saliency mapping with the same shape as input x.
        '''
        pass


class VanillaGradExplainer(Explainer):
    """Regular input gradient explanation."""

    def __init__(self, times_input=False):
        """
        Args:
            times_input: Whether to multiply input as postprocessing.
        """
        self.times_input = times_input

    def explain(self, model, x):
        x = Variable(x, requires_grad=True)
        output = model(x)
        y = output.max(1)[1]
        x_grad = self.get_input_grad(x, output, y).data
        if self.times_input:
            x_grad *= x.data
        return x_grad


class VATExplainer:
    """Explain with the eigenvector with the largest eigenvalue of the hessian
    matrix. Use cross-entropy loss instead of KL divergence as in the original
    paper.

    See https://arxiv.org/abs/1507.00677.
    """

    def __init__(self, xi=1e-6, n_iter=1, times_input=False):
        """
        Args:
            xi:
                hyperparameter of VAT.
            n_iter:
                number of iterations.
            times_input:
                Whether to multiply input as postprocessing.
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
        for _ in range(self.n_iter):
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


class CASO(VanillaGradExplainer):

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
        '''
        Args:
            lambda_t1 (float or torch.cuda.FloatTensor):
                Coefficient of the first Taylor term. Can be different for
                examples in the same batch by passing a Tensor.
            lambda_t2 (float or torch.cuda.FloatTensor):
                Coefficient of the second Taylor term.
            lambda_l1 (float or torch.cuda.FloatTensor):
                Coefficient of the L1-norm term.
            lambda_l2 (float or torch.cuda.FloatTensor):
                Coefficient of the L2-norm term.
            n_iter:
                Number of iterations of optimization.
            optim:
                Type of optimizer (adam or sgd).
            lr:
                Learning rate.
            init:
                Initialization method (zero, random, grad, vat).
            times_input:
                Whether to multiply input as postprocessing.
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
        '''Initialize the delta vector that becomse the saliency.'''
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

    def explain(self, model, x):
        batch_size, n_chs, height, width = x.shape
        x = Variable(x, requires_grad=True)
        delta = self.initialize_delta(model, x)
        if self.optim == 'sgd':
            optimizer = torch.optim.SGD([delta], lr=self.lr, momentum=0.9)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam([delta], lr=self.lr)
        self.history = defaultdict(list)
        for i in range(self.n_iter):
            output = model(x)
            y = output.max(1)[1]
            x_grad = self.get_input_grad(x, output, y, create_graph=True)
            x_grad = x_grad.view((batch_size, n_chs, -1))
            hvp, = torch.autograd.grad(x_grad.dot(delta).sum(), x,
                                       create_graph=True)
            hvp = hvp.view((batch_size, n_chs, -1))
            t1 = x_grad.dot(delta).sum()
            t2 = 0.5 * (delta * hvp)
            l1 = F.l1_loss(delta, torch.zeros_like(delta), reduce=False)
            l2 = F.mse_loss(delta, torch.zeros_like(delta), reduce=False)
            # t1 = t1.sum(2).sum(1) / (n_chs * height * width)
            t2 = t2.sum(2).sum(1) / (n_chs * height * width)
            l1 = l1.sum(2).sum(1) / (n_chs * height * width)
            l2 = l2.sum(2).sum(1) / (n_chs * height * width)
            # t1 = (self.lambda_t1 * t1).sum() / batch_size
            t1 = self.lambda_t1 * t1
            t2 = (self.lambda_t2 * t2).sum() / batch_size
            l1 = (self.lambda_l1 * l1).sum() / batch_size
            l2 = (self.lambda_l2 * l2).sum() / batch_size
            loss = (
                - t1
                - t2
                + l1
                + l2
            )
            # log optimization
            vmax = delta.abs().sum(1).max(1)[0]
            vmin = delta.abs().sum(1).min(1)[0]
            self.history['l1'].append(l1.data.cpu().numpy())
            self.history['l2'].append(l2.data.cpu().numpy())
            self.history['grad'].append(t1.data.cpu().numpy())
            self.history['hessian'].append(t2.data.cpu().numpy())
            self.history['vmax'].append(vmax.data.cpu().numpy())
            self.history['vmin'].append(vmin.data.cpu().numpy())
            # update delta
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        delta = delta.view((batch_size, n_chs, height, width)).data
        if self.times_input:
            delta *= x.data
        return delta


class RobustCASO(CASO):

    '''
    + g^t Delta
    - \lambda_t2 |(\Delta-g/2)^t H g|
    - \lambda_l2 |\Delta|_2^2
    - \lambda_l1 |\Delta|_1
    '''

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
        super(RobustCASO, self).__init__(lambda_t1, lambda_t2,
                                         lambda_l1, lambda_l2,
                                         n_iter, optim, lr, init,
                                         times_input)

    def explain(self, model, x):
        batch_size, n_chs, height, width = x.shape
        x = Variable(x, requires_grad=True)
        delta = self.initialize_delta(model, x)
        if self.optim == 'sgd':
            optimizer = torch.optim.SGD([delta], lr=self.lr, momentum=0.9)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam([delta], lr=self.lr)
        self.history = defaultdict(list)
        for i in range(self.n_iter):
            output = model(x)
            y = output.max(1)[1]
            x_grad = self.get_input_grad(x, output, y, create_graph=True)
            x_grad = x_grad.view((batch_size, n_chs, -1))
            g = x_grad.clone()
            g.detach()
            hvp, = torch.autograd.grad(x_grad.dot(g).sum(), x,
                                       create_graph=True)
            hvp = hvp.view((batch_size, n_chs, -1))
            t1 = x_grad.dot(delta).sum()
            t2 = (delta - g / 2) * hvp
            t2 = F.l1_loss(t2, torch.zeros_like(t2), reduce=False)
            l1 = F.l1_loss(delta, torch.zeros_like(delta), reduce=False)
            l2 = F.mse_loss(delta, torch.zeros_like(delta), reduce=False)

            # t1 = t1.sum(2).sum(1) / (n_chs * height * width)
            t2 = t2.sum(2).sum(1) / (n_chs * height * width)
            l1 = l1.sum(2).sum(1) / (n_chs * height * width)
            l2 = l2.sum(2).sum(1) / (n_chs * height * width)
            # t1 = (self.lambda_t1 * t1).sum() / batch_size
            t1 = self.lambda_t1 * t1
            t2 = (self.lambda_t2 * t2).sum() / batch_size
            l1 = (self.lambda_l1 * l1).sum() / batch_size
            l2 = (self.lambda_l2 * l2).sum() / batch_size
            loss = (
                - t1
                + t2
                + l1
                + l2
            )
            # log optimization
            vmax = delta.abs().sum(1).max(1)[0]
            vmin = delta.abs().sum(1).min(1)[0]
            self.history['l1'].append(l1.data.cpu().numpy())
            self.history['l2'].append(l2.data.cpu().numpy())
            self.history['grad'].append(t1.data.cpu().numpy())
            self.history['hessian'].append(t2.data.cpu().numpy())
            self.history['vmax'].append(vmax.data.cpu().numpy())
            self.history['vmin'].append(vmin.data.cpu().numpy())
            # update delta
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        delta = delta.view((batch_size, n_chs, height, width)).data
        if self.times_input:
            delta *= x.data
        return delta


class IntegrateGradExplainer(VanillaGradExplainer):
    '''Integrated gradient. The final input multiplication is optional.

    See https://arxiv.org/abs/1703.01365.
    '''
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
    '''
    See https://arxiv.org/abs/1706.03825.
    '''

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
    '''Run tuner on the original example to find the best hyperparameters, then
    run the smoothing with fixed hyperparameters.
    '''

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def explain(self, model, x):
        tuner = BatchTuner(CASO, **self.kwargs)
        saliency, lambdas = tuner.explain(model, x, get_lambdas=True)
        lambda_vectors = {k: [] for k in lambdas[0].keys()}
        for key in lambda_vectors:
            ls = [l[key] for l in lambdas]
            lambda_vectors[key] = Variable(torch.FloatTensor(ls).cuda())
        exp = SmoothGradExplainer(CASO(**lambda_vectors))
        return exp.explain(model, x)


class BatchTuner:
    '''Tune the hyperparameter of CASO and its variations.
    For each hyperparameter, form a batch of examples, each with one
    hyperparameter setting.
    '''

    def __init__(self, exp_cls=CASO,
                 n_steps=16, n_iter_search=3,
                 lambda_t1=None, lambda_t2=None,
                 lambda_l1=None, lambda_l2=None,
                 n_iter=10, optim='adam', lr=1e-4, init='zero',
                 times_input=False):
        '''
        Args:
            Exp:
                The explainer class.
            n_step:
                Number of hyperparameters to search at each step.
            n_iter_search:
                Number of iterations of search.
            lambda_t1 (Variable(torch.cuda.FloatTensor) or float or None):
                Search this hyperparameter if None.
            lambda_t2 (Variable(torch.cuda.FloatTensor) or float or None):
                Search this hyperparameter if None.
            lambda_l1 (Variable(torch.cuda.FloatTensor) or float or None):
                Search this hyperparameter if None.
            lambda_l2 (Variable(torch.cuda.FloatTensor) or float or None):
                Search this hyperparameter if None.
            n_iter, optim, lr, init, times_input:
                Parameters for CASO.
        '''
        self.sparse_args = {
            'n_iter': n_iter,
            'optim': optim,
            'lr': lr,
            'init': init,
            'times_input': times_input,
        }
        self.exp_cls = exp_cls
        t1_lo = lambda_t1 if lambda_t1 else 1
        t1_hi = lambda_t1 if lambda_t1 else 1
        t2_lo = lambda_t2 if lambda_t2 else 1
        t2_hi = lambda_t2 if lambda_t2 else 1
        l1_lo = lambda_l1 if lambda_l1 else 1e-1
        l1_hi = lambda_l1 if lambda_l1 else 2e5
        l2_lo = lambda_l2 if lambda_l2 else 1
        l2_hi = lambda_l2 if lambda_l2 else 1e6
        self.tunables = OrderedDict({
            'lambda_t1': (t1_lo, t1_hi),
            'lambda_t2': (t2_lo, t2_hi),
            'lambda_l1': (l1_lo, l1_hi),
            'lambda_l2': (l2_lo, l2_hi),
        })
        self.n_steps = n_steps
        self.n_iter_search = n_iter_search

    def explain_one(self, model, x, quiet=True):
        '''For one example, search for its best hyperparameters.'''
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        tunables = self.tunables.copy()
        best_lambdas = {key: lo for key, (lo, hi) in tunables.items()}
        best_median = 0
        best_saliency = 0
        for i in range(self.n_iter_search):
            xx = x.repeat(self.n_steps, 1, 1, 1)
            for param, (lo, hi) in tunables.items():
                if lo == hi:
                    continue
                ps = np.geomspace(lo, hi, self.n_steps)
                args = self.sparse_args.copy()
                args.update(best_lambdas)
                args[param] = Variable(torch.FloatTensor(ps).cuda())
                model.zero_grad()
                saliency = self.exp_cls(**args).explain(model, xx)
                saliency = saliency.cpu().numpy()
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
        if not quiet:
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


class LambdaTunerExplainer:

    def __init__(self, times_input=False):
        self.times_input = times_input

    def explain(self, model, x, get_lambdas=False, ):
        input_data = x.clone()

        best_median = 0
        lambda_1 = 0
        lambda_2 = 0

        # get initial explanation
        saliency = CASO(lambda_l1=lambda_1,
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
                saliency = CASO(
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

            saliency = CASO(
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
                saliency = CASO(
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

            saliency = CASO(
                lambda_l1=lambda_1,
                lambda_l2=lambda_2).explain(model, x)
            current_median_difference = viz.get_median_difference(
                viz.agg_clip(saliency.cpu().numpy()))
            print('lambda_1', lambda_1, 'lambda_2', lambda_2,
                  'current_median_difference', current_median_difference)

        print("Done Tuning Lambda_L2")
        # because current settings are one too far here
        lambda_2 = lambda_2 / increase_rate

        saliency = CASO(
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
