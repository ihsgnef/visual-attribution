import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, 2, dim=1, keepdim=True) + 1e-8
    return d


def _kl_div(log_probs, probs):
    # pytorch KLDLoss is averaged over all dim if size_average=True
    kld = F.kl_div(log_probs, probs, size_average=False)
    return kld / log_probs.shape[0]


class VATExplainer:

    def __init__(self, model, xi=1e-6, n_iterations=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param n_iterations: number of iterations (default: 1)
        """
        self.model = model
        self.xi = xi
        self.n_iterations = n_iterations

    def explain(self, x, ind=None):
        output = self.model(x)
        ind = output.max(1)[1]
        d_total = torch.zeros(x.shape).cuda()

        #for _ in range(self.n_iterations):
        for _ in range(25):
            d = torch.rand(x.shape).sub(0.5).cuda()
            d = _l2_normalize(d)
            self.model.zero_grad()
            d = Variable(self.xi * d, requires_grad=True)
            pred_hat = self.model(x + d)
            adv_loss = F.cross_entropy(pred_hat, ind)
            d_grad, = torch.autograd.grad(adv_loss, d)
            d = _l2_normalize(d_grad.data)
            d_total = d_total + d


