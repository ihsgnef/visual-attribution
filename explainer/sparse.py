import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseExplainer(object):
    def __init__(self, model, lambda_t2=1, lambda_l1=1e4, lambda_l2=1e5,
                 n_iterations=10):
        self.model = model        
        self.lambda_t2 = lambda_t2 
        self.lambda_l1 = lambda_l1 
        self.lambda_l2 = lambda_l2 
        self.n_iterations = n_iterations
    
    def explain(self, inp, ind=None):
        batch_size, n_chs, img_width, img_height = inp.shape
        delta = torch.zeros((batch_size, n_chs, img_width * img_height)).cuda()
        delta = nn.Parameter(delta, requires_grad=True)
        # optimizer = torch.optim.SGD([delta], lr=0.1)
        optimizer = torch.optim.Adam([delta], lr=0.0001)
        for i in range(self.n_iterations):
            output = self.model(inp)
            # if ind is None:
            ind = output.max(1)[1]
            out_loss = F.cross_entropy(output, ind)
            inp_grad, = torch.autograd.grad(out_loss, inp, create_graph=True)
            inp_grad = inp_grad.view((batch_size, n_chs, img_width * img_height))
            hessian_delta_vp, = torch.autograd.grad(
                    inp_grad.dot(delta).sum(), inp, create_graph=True)
            hessian_delta_vp = hessian_delta_vp.view((batch_size, n_chs, img_width * img_height))
            taylor_1 = inp_grad.dot(delta).sum()
            taylor_2 = 0.5 * delta.dot(hessian_delta_vp).sum()
            l1_term = F.l1_loss(delta, torch.zeros_like(delta))
            l2_term = F.mse_loss(delta, torch.zeros_like(delta))
            print(taylor_1.data.cpu().numpy(), taylor_2.data.cpu().numpy(),
                  l1_term.data.cpu().numpy(), l2_term.data.cpu().numpy())
            loss = - taylor_1 - self.lambda_t2 * taylor_2
            loss += self.lambda_l1 * l1_term + self.lambda_l2 * l2_term
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        delta = delta.view((batch_size, n_chs, img_width, img_height))
        return delta.data
