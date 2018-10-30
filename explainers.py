import numpy as np
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import viz


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, 2, dim=1, keepdim=True) + 1e-8
    return d


def _kl_div(log_probs, probs):
    kld = F.kl_div(log_probs, probs, reduction='sum')
    return kld / log_probs.shape[0]


class Explainer:

    def get_input_grad(self, x, output, y, create_graph=False,
                       cross_entropy=True):
        '''Compute gradient of loss w.r.t input x.
        Args:
            x (torch.Tensor):
                Input tensor.
            output (torch.Tensor):
                Output before softmax.
            y (torch.Tensor):
                Class label.
            create_graph:
                Set to True if higher-order gradient will be used.
            cross_entropy:
                Use by default the cross entropy loss.
        Rerturns:
            torch.Tensor:
                The gradient, shape identical to x.
        '''
        if cross_entropy:
            loss = F.cross_entropy(output, y)
            x_grad, = torch.autograd.grad(loss, x, create_graph=create_graph)
        else:
            grad_out = torch.zeros_like(output)
            grad_out.scatter_(1, y.unsqueeze(0).t(), 1.0)
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
        # TODO add optional y
        pass


class VanillaGrad(Explainer):
    """Regular input gradient explanation."""

    def __init__(self, times_input=False):
        """
        Args:
            times_input: Whether to multiply input as postprocessing.
        """
        self.times_input = times_input

    def explain(self, model, x):
        x.requires_grad_()
        output = model(x)
        y = output.max(1)[1]
        x_grad = self.get_input_grad(x, output, y)
        if self.times_input:
            x_grad *= x
        return x_grad.detach()


class VATExplainer:
    """Explain with the eigenvector with the largest eigenvalue of the hessian
    matrix. Use cross-entropy loss instead of KL divergence as in the original
    paper.

    See https://arxiv.org/abs/1507.00677.
    """

    def __init__(self, xi=1e-6, n_iter=10, times_input=False):
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
        self.n_iter = n_iter
        self.times_input = times_input

    def explain(self, model, x, KL=True):
        logits = model(x)
        y = logits.max(1)[1]
        if KL:
            p_0 = F.softmax(logits, dim=1).detach()
        delta = torch.rand(x.shape).sub(0.5).cuda()
        delta = _l2_normalize(delta)
        delta.requires_grad_()
        for _ in range(self.n_iter):
            model.zero_grad()
            delta = self.xi * delta
            logits_hat = model(x + delta)
            if KL:
                p_hat = F.softmax(logits_hat, dim=1)
                adv_loss = _kl_div(p_hat, p_0)
            else:
                adv_loss = F.cross_entropy(F.log_softmax(logits_hat), y)
            d_grad, = torch.autograd.grad(adv_loss, delta)
            delta = _l2_normalize(d_grad)
            delta.requires_grad_()
        if self.times_input:
            delta *= x
        return delta.detach()


def get_eigen_value(model, x, n_iter=10):
    x.requires_grad_()
    batch_size, n_chs, height, width = x.shape
    delta = torch.rand(batch_size, n_chs * height * width)
    delta = _l2_normalize(delta.sub(0.5)).cuda()
    for i in range(n_iter):
        model.zero_grad()
        logits = model(x)
        y = logits.max(1)[1]
        loss = F.cross_entropy(logits, y)
        x_grad, = torch.autograd.grad(loss, x, create_graph=True)
        x_grad = x_grad.view(batch_size, -1)
        hvp, = torch.autograd.grad((x_grad * delta).sum(), x)
        hvp = hvp.view(batch_size, -1).detach()
        ev = (delta * hvp).sum(1)
        delta = _l2_normalize(hvp)
        # print('Power Method Eigenvalue Iteration',
        #       '{}: {}'.format(i, ev.tolist()))
    return ev, delta


class CASO(Explainer):

    def __init__(self,
                 lambda_t1=1,
                 lambda_t2=1,
                 lambda_l1=1e4,
                 lambda_l2=1e5,
                 n_iter=10,
                 optim='adam',
                 lr=1e-3,
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
        assert init in ['zero', 'random', 'grad', 'eig']
        self.history = defaultdict(list)

    def initialize_delta(self, model, x):
        '''Initialize the delta vector that becomse the saliency.'''
        batch_size, n_chs, height, width = x.shape
        if self.init == 'zero':
            delta = torch.zeros((batch_size, n_chs, height * width)).cuda()
        elif self.init == 'grad':
            output = model(x)
            y = output.max(1)[1]
            delta = self.get_input_grad(x, output, y).detach()
        elif self.init == 'random':
            delta = torch.rand(x.shape)
            delta = delta.sub(0.5).cuda()
            delta = _l2_normalize(delta)
        elif self.init == 'eig':
            delta = VATExplainer().explain(model, x)
        delta = delta.view(batch_size, -1)
        delta = nn.Parameter(delta, requires_grad=True)
        return delta

    def explain(self, model, x):
        x.requires_grad_()
        batch_size, n_chs, height, width = x.shape
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
            x_grad = x_grad.view((batch_size, -1))
            hvp, = torch.autograd.grad((x_grad * delta).sum(), x,
                                       create_graph=True)
            hvp = hvp.view((batch_size, -1))
            t1 = (x_grad * delta)
            t2 = 0.5 * (delta * hvp)
            l1 = F.l1_loss(delta, torch.zeros_like(delta), reduction='none')
            l2 = F.mse_loss(delta, torch.zeros_like(delta), reduction='none')
            t1 = (self.lambda_t1 * t1).sum() / x.nelement()
            t2 = (self.lambda_t2 * t2).sum() / x.nelement()
            l1 = (self.lambda_l1 * l1).sum() / x.nelement()
            l2 = (self.lambda_l2 * l2).sum() / x.nelement()
            loss = (
                - t1
                # - t2
                + l1
                # + l2
            )
            print('{}\t{}\t{}\t{}'.format(
                t1.detach().cpu().numpy(),
                t2.detach().cpu().numpy(),
                l1.detach().cpu().numpy(),
                l2.detach().cpu().numpy(),
            ))
            # # log optimization
            # self.history['l1'].append(l1.data.cpu().numpy())
            # self.history['l2'].append(l2.data.cpu().numpy())
            # self.history['grad'].append(t1.data.cpu().numpy())
            # self.history['hessian'].append(t2.data.cpu().numpy())
            # update delta
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            delta = _l2_normalize(delta.detach())
        print()
        delta = delta.view((batch_size, n_chs, height, width))
        if self.times_input:
            delta *= x
        return delta.detach()


# class RobustCASO(CASO):
#
#     '''
#     + g^t Delta
#     - \lambda_t2 |(\Delta-g/2)^t H g|
#     - \lambda_l2 |\Delta|_2^2
#     - \lambda_l1 |\Delta|_1
#     '''
#
#     def __init__(self,
#                  lambda_t1=1,
#                  lambda_t2=1,
#                  lambda_l1=1e4,
#                  lambda_l2=1e5,
#                  n_iter=10,
#                  optim='adam',
#                  lr=1e-4,
#                  init='zero',
#                  times_input=False,
#                  ):
#         super(RobustCASO, self).__init__(lambda_t1, lambda_t2,
#                                          lambda_l1, lambda_l2,
#                                          n_iter, optim, lr, init,
#                                          times_input)
#
#     def explain(self, model, x):
#         x.requires_grad_()
#         batch_size, n_chs, height, width = x.shape
#         delta = self.initialize_delta(model, x)
#         if self.optim == 'sgd':
#             optimizer = torch.optim.SGD([delta], lr=self.lr, momentum=0.9)
#         elif self.optim == 'adam':
#             optimizer = torch.optim.Adam([delta], lr=self.lr)
#         self.history = defaultdict(list)
#         for i in range(self.n_iter):
#             output = model(x)
#             y = output.max(1)[1]
#             x_grad = self.get_input_grad(x, output, y, create_graph=True)
#             x_grad = x_grad.view((batch_size, -1))
#             g = x_grad.detach().clone()
#             hvp, = torch.autograd.grad(x_grad.dot(g).sum(), x,
#                                        create_graph=True)
#             hvp = hvp.view((batch_size, -1))
#             t1 = (x_grad * delta).sum()
#             t2 = (delta - g / 2) * hvp
#             t2 = F.l1_loss(t2, torch.zeros_like(t2), reduce=False)
#             l1 = F.l1_loss(delta, torch.zeros_like(delta), reduce=False)
#             l2 = F.mse_loss(delta, torch.zeros_like(delta), reduce=False)
#             # t1 = t1.sum(2).sum(1) / (n_chs * height * width)
#             t2 = t2.sum(2).sum(1) / (n_chs * height * width)
#             l1 = l1.sum(2).sum(1) / (n_chs * height * width)
#             l2 = l2.sum(2).sum(1) / (n_chs * height * width)
#             # t1 = (self.lambda_t1 * t1).sum() / batch_size
#             t1 = self.lambda_t1 * t1
#             t2 = (self.lambda_t2 * t2).sum() / batch_size
#             l1 = (self.lambda_l1 * l1).sum() / batch_size
#             l2 = (self.lambda_l2 * l2).sum() / batch_size
#             loss = (
#                 - t1
#                 + t2
#                 + l1
#                 + l2
#             )
#             # log optimization
#             vmax = delta.abs().sum(1).max(1)[0]
#             vmin = delta.abs().sum(1).min(1)[0]
#             self.history['l1'].append(l1.data.cpu().numpy())
#             self.history['l2'].append(l2.data.cpu().numpy())
#             self.history['grad'].append(t1.data.cpu().numpy())
#             self.history['hessian'].append(t2.data.cpu().numpy())
#             self.history['vmax'].append(vmax.data.cpu().numpy())
#             self.history['vmin'].append(vmin.data.cpu().numpy())
#             # update delta
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         delta = delta.view((batch_size, n_chs, height, width)).data
#         if self.times_input:
#             delta *= x.data
#         return delta


class IntegrateGrad(Explainer):
    '''Integrated gradient. The final input multiplication is optional.

    See https://arxiv.org/abs/1703.01365.
    '''
    def __init__(self, n_iter=100, times_input=False):
        self.n_iter = n_iter
        self.times_input = times_input

    def explain(self, model, x):
        x.requires_grad_()
        delta = 0
        for alpha in np.arange(1 / self.n_iter, 1.0, 1 / self.n_iter):
            logits = model(x * alpha)
            y = logits.max(1)[1]
            g = self.get_input_grad(x, logits, y) * alpha
            delta += g.detach()
        delta = delta / self.n_iter
        if self.times_input:
            delta *= x
        return delta


class SmoothGrad:
    '''
    See https://arxiv.org/abs/1706.03825.
    '''

    def __init__(self, base_explainer=None, stdev_spread=0.15,
                 nsamples=25, magnitude=True, times_input=False):
        if base_explainer is None:
            base_explainer = VanillaGrad()
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
            lambda_vectors[key] = torch.FloatTensor(ls).cuda()
        exp = SmoothGrad(CASO(**lambda_vectors))
        return exp.explain(model, x)


class EigenCASO(CASO):
    '''Set lambda_l2 based on the largest eigenvalue of the input hessian'''
    
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
        super(EigenCASO, self).__init__(lambda_t1, lambda_t2,
                                        lambda_l1, lambda_l2,
                                        n_iter, optim, lr, init,
                                        times_input)

    def explain(self, model, x):
        batch_size, n_chs, height, width = x.shape

        ev, delta = get_eigen_value(model, x)
        output = model(x)
        y = output.max(1)[1]
        x_grad = self.get_input_grad(x, output, y, create_graph=True)
        x_grad = x_grad.view((batch_size, -1))
        hvp_eigen, = torch.autograd.grad((x_grad * delta).sum(), x,
                                         create_graph=True)
        hvp_eigen = hvp_eigen.view((batch_size, -1))

        model.zero_grad()
        output = model(x)
        y = output.max(1)[1]
        x_grad = self.get_input_grad(x, output, y, create_graph=True)
        x_grad = x_grad.view((batch_size, -1))
        grad = _l2_normalize(x_grad.detach())
        hvp_grad, = torch.autograd.grad((x_grad * grad).sum(), x,
                                        create_graph=True)
        hvp_grad = hvp_grad.view((batch_size, -1))

        ev = ev.detach().cpu().numpy()
        hvp_eigen = hvp_eigen.detach().cpu().numpy()
        hvp_grad = hvp_grad.detach().cpu().numpy()

        print('ev: {:.4f}\thvp_e: {:.4f}\thvp_g: {:.4f}'.format(
            ev.tolist()[0], 
            np.linalg.norm(hvp_eigen, 2).tolist(),
            np.linalg.norm(hvp_grad, 2).tolist(),
        ))
        return VanillaGrad().explain(model, x)


class BatchTuner:
    '''Tune the hyperparameter of CASO and its variations.
    For each hyperparameter, form a batch of examples, each with one
    hyperparameter setting.
    '''

    def __init__(self, explainer_class=CASO,
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
            lambda_t1 (torch.cuda.FloatTensor or float or None):
            lambda_t2 (torch.cuda.FloatTensor or float or None):
            lambda_l1 (torch.cuda.FloatTensor or float or None):
            lambda_l2 (torch.cuda.FloatTensor or float or None):
                Search hyperparameter if None.
            n_iter, optim, lr, init, times_input:
                Parameters for CASO.
        '''
        self.caso_args = {
            'n_iter': n_iter,
            'optim': optim,
            'lr': lr,
            'init': init,
            'times_input': times_input,
        }
        self.explainer_class = explainer_class

        # turn of tuning by setting upper and lower bound to the same value
        t1_lo = t1_hi = lambda_t1 if lambda_t1 else 1.
        t2_lo = t2_hi = lambda_t2 if lambda_t2 else 1.
        l1_lo = lambda_l1 if lambda_l1 else 1e-1
        l1_hi = lambda_l1 if lambda_l1 else 2e5
        l2_lo = lambda_l2 if lambda_l2 else 1.
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
                ps = np.geomspace(lo, hi, self.n_steps, dtype=np.float32)
                args = self.caso_args.copy()
                args.update(best_lambdas)
                args[param] = torch.FloatTensor(ps[:, np.newaxis]).cuda()
                model.zero_grad()
                saliency = self.explainer_class(**args).explain(model, xx)
                saliency = saliency.cpu().numpy()
                s = viz.agg_clip(saliency)
                medians = [viz.get_median_difference(x) for x in s]
                best_idx = np.argmax(medians)
                ps = ps.tolist()
                print()
                print(i, param)
                print(' '.join('{:.3f}'.format(x) for x in ps))
                print(' '.join('{:.3f}'.format(x) for x in medians))
                print()
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


class NewExplainer(Explainer):

    def explain(self, model, x, y=None):
        x_data = x.clone()
        y_hat = model(x)
        ws = []
        for yh in y_hat[0]:
            model.zero_grad()
            x_grad, = torch.autograd.grad(yh, x, retain_graph=True)
            ws.append(x_grad.data[0])
        # classes, channel, height, width
        model.zero_grad()
        W = torch.stack(ws, -1)
        n_chs, height, width, n_cls = W.shape
        W = W.view(-1, n_cls)

        y_prob = F.softmax(y_hat, 1).data  # 1, classes

        W = W.cpu()
        y_prob = y_prob.cpu()

        D = torch.diag(y_prob[0])
        A = (D - y_prob.transpose(0, 1).mm(y_prob))

        sigma_A, U_A = torch.symeig(A, eigenvectors=True)

        sigma_A_sqrt = torch.sqrt(sigma_A)
        sigma_A_sqrt = torch.diag(sigma_A_sqrt)
        B = W.mm(U_A)
        B = B.mm(sigma_A_sqrt)

        BTB = B.transpose(0, 1).mm(B)
        sigma_B_sq, V_B = torch.symeig(BTB, eigenvectors=True)
        rank = np.linalg.matrix_rank(BTB)

        # reverse order of sigma
        # inv_idx = torch.arange(sigma_B_sq.size(0)-1, -1, -1).long()
        # sigma_B_sq = sigma_B_sq.index_select(0, inv_idx)

        # print('rank', rank)
        # zero out lower eigenvalues
        for index in range(n_cls - rank):
            sigma_B_sq[index] = 0.0
            V_B[index] = 0.0

        # print('Our Method Eigenvalues', sigma_B_sq.numpy().tolist())

        sigma_B_inv = torch.rsqrt(sigma_B_sq)

        for index in range(n_cls - rank):
            sigma_B_inv[index] = 0.0 # remove smallest eigenvectors because rank is c - 1

        sigma_B_inv = torch.diag(sigma_B_inv)

        HEV = V_B.mm(sigma_B_inv)
        HEV = B.mm(HEV)
        # print('Our Method Eigenvectors', HEV)

        # inverse
        recip = torch.reciprocal(sigma_B_sq)
        for index in range(n_cls - rank):
            recip[index] = 0.0 # remove smallest eigenvectors because rank is c - 1

        output = model(x)
        y = output.max(1)[1]
        x_grad = self.get_input_grad(x, output, y).cpu().data
        x_grad = x_grad.view(-1, 1)
        newtons = HEV.transpose(0, 1).mm(x_grad)

        recip = torch.diag(torch.reciprocal(sigma_B_sq))
        temp = HEV.mm(recip)
        newtons = temp.mm(newtons)

        delta = -1 * newtons
        batch_size = 1
        delta = delta.view((batch_size, n_chs, height, width))
        return delta
