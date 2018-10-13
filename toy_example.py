import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
from collections import defaultdict

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, 2, dim=1, keepdim=True) + 1e-8
    return d


class Explainer:

    def get_input_grad(self, x, output, y, create_graph=False,
                       cross_entropy=True):  
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
        pass

class CASO(Explainer):

    def __init__(self,
                 lambda_t1=0,
                 lambda_t2=1,
                 lambda_l1=0,
                 lambda_l2=0,
                 n_iter=100,
                 optim='adam',
                 lr=1e-3,
                 init='zero',
                 times_input=False,
                 ):        
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
        # batch_size, n_chs, height, width = x.shape
        # if self.init == 'zero':
        #     delta = torch.zeros((batch_size, n_chs, height * width)).cuda()
        # elif self.init == 'grad':
        #     output = model(x)
        #     y = output.max(1)[1]
        #     delta = self.get_input_grad(x, output, y).data
        #     delta = delta.view(batch_size, n_chs, -1)
        # elif self.init == 'random':
        #     delta = torch.rand((batch_size, n_chs, height * width))
        #     delta = delta.sub(0.5).cuda()
        #     delta = _l2_normalize(delta)
        # elif self.init == 'vat':
        #     delta = VATExplainer().explain(model, x.data)
        #     delta = delta.view(batch_size, n_chs, height * width)
        
        batch_size, inp_size = x.shape
        # delta = torch.zeros(batch_size, inp_size)
        delta = torch.rand((batch_size, inp_size))
        delta = delta.sub(0.5) / 100
        # delta = _l2_normalize(delta) 
        delta = nn.Parameter(delta, requires_grad=True)
        return delta

    def explain(self, model, x):
        # batch_size, n_chs, height, width = x.shape
        batch_size, inp_size = x.shape
        x = Variable(x, requires_grad=True)
        delta = self.initialize_delta(model, x)
        if self.optim == 'sgd':
            optimizer = torch.optim.SGD([delta], lr=self.lr, momentum=0.9)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam([delta], lr=self.lr)
        # self.history = defaultdict(list)
        for i in range(self.n_iter):
            output = model(x)
            y = output.max(1)[1]
            x_grad = self.get_input_grad(x, output, y, create_graph=True)
            # x_grad = x_grad.view((batch_size, n_chs, -1))
            x_grad = x_grad.view((batch_size, -1))
            hvp, = torch.autograd.grad(x_grad.dot(delta).sum(), x,
                                       create_graph=True)
            # hvp = hvp.view((batch_size, n_chs, -1))
            hvp = hvp.view((batch_size, -1))
            t1 = x_grad.dot(delta).sum()
            t2 = delta * hvp # 0.5 * (delta * hvp)
            l1 = F.l1_loss(delta, torch.zeros_like(delta), reduce=False)
            l2 = F.mse_loss(delta, torch.zeros_like(delta), reduce=False)            
            # t2 = t2.sum(2).sum(1) / (n_chs * height * width)
            # l1 = l1.sum(2).sum(1) / (n_chs * height * width)
            # l2 = l2.sum(2).sum(1) / (n_chs * height * width)                                                
            t2 = t2.sum(1) / (inp_size)                                
            l1 = l1.sum(1) / (inp_size)
            l2 = l2.sum(1) / (inp_size)            
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
            # vmax = delta.abs().sum(1).max(1)[0]
            # vmin = delta.abs().sum(1).min(1)[0]
            # self.history['l1'].append(l1.data.cpu().numpy())
            # self.history['l2'].append(l2.data.cpu().numpy())
            # self.history['grad'].append(t1.data.cpu().numpy())
            # self.history['hessian'].append(t2.data.cpu().numpy())
            # self.history['vmax'].append(vmax.data.cpu().numpy())
            # self.history['vmin'].append(vmin.data.cpu().numpy())
            # update delta
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # delta = delta.view((batch_size, n_chs, height, width)).data        
        
        print(delta)
        delta = _l2_normalize(delta)                
        print(delta)        
        delta = delta.view((batch_size, inp_size)).data
        if self.times_input:
            delta *= x.data
        return delta

def power_method(model, x):    
    #batch_size, n_chs, height, width = x.shape
    batch_size, inp_size = x.shape
    x_data = x.clone()
    x = Variable(x, requires_grad=True)
    #d = torch.rand(batch_size, n_chs, height * width)
    d = torch.rand(batch_size, inp_size)
    #d = _l2_normalize(d.sub(0.5)).cuda()
    d = _l2_normalize(d.sub(0.5))

    for iterat in range(10):
        model.zero_grad()
        output = model(x)
        y = output.max(1)[1]
        loss = F.cross_entropy(output, y)
        x_grad, = torch.autograd.grad(loss, x, create_graph=True)
        #x_grad = x_grad.view(batch_size, n_chs, -1)
        x_grad = x_grad.view(batch_size, -1)
        d_var = Variable(d, requires_grad=True)
        hvp, = torch.autograd.grad(x_grad.dot(d_var).sum(), x)
        #hvp = hvp.data.view(batch_size, n_chs, -1)
        hvp = hvp.data.view(batch_size, -1)
        taylor_2 = (d * hvp).sum()
        #d = _l2_normalize(hvp).view(batch_size, n_chs, -1)
        d = _l2_normalize(hvp).view(batch_size, -1)
        print('Power Method Eigenvalue Iteration', iterat, ':', taylor_2)    
    print('Power Method Eigenvector', d)    

def our_hessian(model, x):
    x_data = x.clone()
    x = Variable(x, requires_grad=True)
    y_hat = model(x)
    ws = []
    for yh in y_hat[0]:
        model.zero_grad()
        x_grad, = torch.autograd.grad(yh, x, retain_graph=True)
        ws.append(x_grad.data[0])
    # classes, channel, height, width
    model.zero_grad()
    W = torch.stack(ws, -1)
    ######n_chs, height, width, n_cls = W.shape    
    size, n_cls = W.shape    
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

    print('rank', rank)
    # zero out lower eigenvalues
    for index in range(n_cls - rank):                
        sigma_B_sq[index] = 0.0        
        V_B[index] = 0.0    

    print('Our Method Eigenvalues', sigma_B_sq.numpy().tolist())    
            
    sigma_B_inv = torch.rsqrt(sigma_B_sq)        
    
    for index in range(n_cls - rank):
        sigma_B_inv[index] = 0.0 # remove smallest eigenvectors because rank is c - 1            

    sigma_B_inv = torch.diag(sigma_B_inv)
    print(sigma_B_inv)

    HEV = V_B.mm(sigma_B_inv)
    HEV = B.mm(HEV)
    print('Our Method Eigenvectors', HEV)

    # inverse
    recip = torch.reciprocal(sigma_B_sq)
    for index in range(n_cls - rank):
        recip[index] = 0.0 # remove smallest eigenvectors because rank is c - 1                    

    print(torch.diag(sigma_B_sq).mm(torch.diag(recip)))
    print(HEV.transpose(0,1).mm(HEV))    
    recip = torch.diag(recip)            
    Hessian_inverse = HEV.mm(recip)
    Hessian_inverse = Hessian_inverse.mm(HEV.transpose(0, 1))

    return B.mm(B.transpose(0,1)), Hessian_inverse

def exact_hessian(loss_grad, input):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)    
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = autograd.grad(g_vector[idx], input, create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1        
        hessian[idx] = g2.data
    return hessian.cpu().numpy()    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()      
        self.fc1 = nn.Linear(5, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 2)

    def forward(self, x):                
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))        
        return self.fc5(x)            

net = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

fake_input = Variable(torch.randn(1,5), requires_grad=True)
target = Variable(torch.LongTensor(1).random_(0, 1))  # a dummy target, for example
output = net(fake_input)
loss = criterion(output, target)

power_method(net, fake_input.data)
exact_hessian = exact_hessian(autograd.grad(loss, fake_input, create_graph=True), fake_input)
our_hessian, Hessian_inverse = our_hessian(net, fake_input.data)

# print("Full Hessian Using Our Method")
# print(our_hessian.numpy())
# print("Exact Hessian Using PyTorch")
# print(exact_hessian)

# print(our_hessian.mm(Hessian_inverse))
# print(Hessian_inverse.mm(our_hessian))

# print(np.matmul(our_hessian.numpy(), np.linalg.pinv(our_hessian.numpy())))
# print(np.matmul(np.linalg.pinv(our_hessian.numpy()), our_hessian.numpy()))

our_hessian = our_hessian
a = our_hessian
B = np.linalg.pinv(our_hessian.numpy())
print(np.dot(B,a))
assert(np.allclose(a, np.dot(a, np.dot(B, a))))

# explainer = CASO()
# delta = explainer.explain(net, fake_input.data)
# assert(np.allclose(exact_hessian, our_hessian, rtol=1e-05, atol=1e-03))

