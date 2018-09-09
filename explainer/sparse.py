import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class SparseExplainer(object):
    def __init__(self, model, hessian_coefficient=1,
                 lambda_l1=1e3, lambda_l2=0,
                 n_iterations=10):
        self.model = model
        self.hessian_coefficient = hessian_coefficient
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.n_iterations = n_iterations
        print(self.lambda_l1, self.lambda_l2)

    def explain(self, inp, ind=None, return_loss=False):
        batch_size, n_chs, img_width, img_height = inp.shape
        img_size = img_width * img_height
        delta = torch.zeros((batch_size, n_chs, img_size)).cuda()
        delta = nn.Parameter(delta, requires_grad=True)
        optimizer = torch.optim.SGD([delta], lr=0.1)
        # optimizer = torch.optim.Adam([delta], lr=0.0001)
        loss_history = defaultdict(list)
        for i in range(self.n_iterations):
            output = self.model(inp)
            # if ind is None:
            ind = output.max(1)[1]
            # TODO, find a way to not recalculate grad each time
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
            loss = - taylor_1 - self.hessian_coefficient * taylor_2
            loss += self.lambda_l1 * l1_term + self.lambda_l2 * l2_term            
            if i != 0:
                loss_history['l1'].extend(self.lambda_l1 * l1_term)
                loss_history['l2'].extend(self.lambda_l2 * l2_term)
                loss_history['grad'].extend(- taylor_1)
                loss_history['hessian'].extend(- self.hessian_coefficient * taylor_2)
            else:
                loss_history['l1'] = [self.lambda_l1 * l1_term]
                loss_history['l2'] = [self.lambda_l2 * l2_term]
                loss_history['grad'] = [- taylor_1]
                loss_history['hessian'] = [- self.hessian_coefficient * taylor_2]
            # print(taylor_1.data.cpu().numpy(), taylor_2.data.cpu().numpy(),
            #       l1_term.data.cpu().numpy(), l2_term.data.cpu().numpy())
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        delta = delta.view((batch_size, n_chs, img_width, img_height))
        if return_loss:
            return delta.data.abs(), loss_history
        return delta.data.abs()    # abs as recommended by sohiel
