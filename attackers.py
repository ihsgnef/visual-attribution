import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def get_topk_mask(saliency, k=1e4, flip=False):
    '''Generate binary mask based on saliency value.
    Args:
        saliency (numpy.array):
            3 or 4-dimensional, aggregated saliency.
        k:
            Number of pixels to attack, channel independent if mask is 4D.
        flip:
            If True high value positions get 0 instead of 1.
    Returns:
        numpy.array:
            The mask with shape identical to saliency (3D or 4D).
    '''
    batch_size, n_chs, height, width = saliency.shape
    if len(saliency.shape) == 4:
        saliency = saliency.reshape(batch_size, n_chs, -1)
        topk_mask = np.ones_like(saliency) if flip else np.zeros_like(saliency)
        topk_idx = np.argsort(-saliency, axis=2)[:, :, :k]
        for i in range(batch_size):
            for j in range(n_chs):
                topk_mask[i, j, topk_idx[i, j]] = 0 if flip else 1
        topk_mask = topk_mask.reshape(batch_size, n_chs, height, width)
    elif len(saliency.shape) == 3:
        saliency = saliency.reshape(batch_size, -1)
        topk_mask = np.ones_like(saliency) if flip else np.zeros_like(saliency)
        topk_idx = np.argsort(-saliency, axis=1)[:, :k]
        for i in range(batch_size):
            topk_mask[i, topk_idx[i]] = 0 if flip else 1
        topk_mask = topk_mask.reshape(batch_size, height, width)
        topk_mask = np.expand_dims(topk_mask, axis=1)
        topk_mask = np.tile(topk_mask, (1, n_chs, 1, 1))
    else:
        print('saliency shape wrong')
    return topk_mask


class Attacker:

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

    def explain(self, model, x, saliency=None):
        '''Generate adversarial perturbation for input x.
            Attackers share the general API with Explainers, except for some
            attacks which target specific saliency mapping.
        Args:
            model (torch.nn.Module):
                The model to explain.
            x (torch.cuda.FloatTensor):
                Input tensor.
            saliency (torch.cuda.LongTensor or None):
                Saliency with shape identical to x.
        Returns
            torch.cuda.FloatTensor:
                Perturbation with shape identical to input x.
        '''
        pass


class EmptyAttack(Attacker):
    '''Fake attack to return unperturbed input.'''

    def explain(self, model, x, saliency=None):
        return x


class GhorbaniAttack(Attacker):
    '''
    See https://arxiv.org/abs/1710.10547.
    '''

    def __init__(self, epsilon=2/255, n_iter=20, k=1e4,
                 topk_agg=lambda x: np.abs(x)):
        '''
        Args:
            epsilon
            n_iter:
                Number of iterations
            k:
                The B constant of Ghorbani attack: number of pixels to dampen.
            topk_agg:
                Aggregation function of saliency to select top-k pixels.
        '''
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.k = int(k)
        self.topk_agg = topk_agg

    def explain(self, model, x, saliency=None):
        '''Generate attack against specified saliency mapping.
            If saliency is not specified, assume vanila gradient saliency.
        '''
        batch_size, n_chs, height, width = x.shape
        x_prev = x.clone()
        delta = torch.zeros_like(x)
        y_org = model(Variable(x)).max(1)[1].data
        if saliency is None:
            x_curr = Variable(x, requires_grad=True)
            output = model(x_curr)
            y = output.max(1)[1]
            saliency = self.get_input_grad(x_curr, output, y)
            saliency = saliency.data.cpu().numpy()
        topk_mask = get_topk_mask(self.topk_agg(saliency), self.k)
        topk_mask = torch.FloatTensor(topk_mask).cuda()
        topk_mask = Variable(topk_mask)
        step_size = self.epsilon / self.n_iter
        stopped = [False for _ in range(batch_size)]
        for i in range(self.n_iter):
            model.zero_grad()
            x_curr = Variable(x, requires_grad=True)
            output = model(x_curr)
            y = output.max(1)[1]
            x_grad = self.get_input_grad(x_curr, output, y, create_graph=True)
            topk = (x_grad.abs() * topk_mask).sum()
            delta, = torch.autograd.grad(-topk, x_curr)
            delta = delta.sign().data
            # verify same prediction
            for idx in range(batch_size):
                if stopped[idx] or y.data[idx] != y_org[idx]:
                    x[idx] = x_prev[idx]
                    delta[idx].zero_()
                    stopped[idx] = True
            if all(stopped):
                break
            x_prev = x.clone()
            x = torch.clamp(x + step_size * delta, 0, 1)
        # final check that all predictions remain
        y = model(Variable(x)).max(1)[1].data
        assert (y == y_org).all()
        return x


class ScaledNoiseAttack(Attacker):
    '''Random noise with magnitude scaled based on pixel value.'''

    def __init__(self, epsilon=2 / 255):
        self.epsilon = epsilon

    def explain(self, model, x, saliency=None):
        x = x.cpu().numpy()
        noise = 2 * np.random.randint(2, size=x.shape) - 1
        noise = np.sign(noise) * self.epsilon
        x = np.clip(x + noise * x, 0, 1)
        x = torch.FloatTensor(x).cuda()
        return x


class FGSM(Attacker):
    '''Iterative Fast Gradient Sign Method.'''

    def __init__(self, epsilon=2 / 255, n_iter=10):
        self.epsilon = epsilon
        self.n_iter = n_iter

    def explain(self, model, x, saliency=None):
        batch_size, n_chs, height, width = x.shape
        step_size = self.epsilon / self.n_iter
        for i in range(self.n_iter):
            model.zero_grad()
            x_curr = Variable(x, requires_grad=True)
            output = model(x_curr)
            y = output.max(1)[1]
            x_grad = self.get_input_grad(x, output, y).data
            x_grad = x_grad.sign()
            x = torch.clamp(x + step_size * x_grad, 0, 1)
        return x
