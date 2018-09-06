import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseExplainer(object):
    def __init__(self, model, lambda_t2=0, lambda_l1=0, lambda_l2=0,
                 n_iterations=10):
        self.model = model        
        self.lambda_t2 = lambda_t2 
        self.lambda_l1 = lambda_l1 
        self.lambda_l2 = lambda_l2 
        self.n_iterations = n_iterations
    
    def explain(self, inp, ind=None):
        delta = torch.zeros_like(inp.data).cuda()
        delta = nn.Parameter(delta, requires_grad=True)
        optimizer = torch.optim.SGD([delta], lr=0.1)
        for i in range(self.n_iterations):
            output = self.model(inp)
            # if ind is None:
            ind = output.max(1)[1]
            out_loss = F.cross_entropy(output, ind)
            inp_grad, = torch.autograd.grad(out_loss, inp, create_graph=True)
            hessian_delta_vp, = torch.autograd.grad(
                    (inp_grad @ delta).sum(), inp, create_graph=True)
            taylor_1 = (inp_grad @ delta).sum()
            taylor_2 = 0.5 * (delta.t() @ hessian_delta_vp).sum()
            l1_term = F.l1_loss(delta, torch.zeros_like(delta))
            l2_term = F.mse_loss(delta, torch.zeros_like(delta))
            loss = - taylor_1 - self.lambda_t2 * taylor_2
            loss += self.lambda_l1 * l1_term + self.lambda_l2 * l2_term
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return delta.data
