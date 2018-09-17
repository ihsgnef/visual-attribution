from torchvision import models
from torch.autograd import Variable
from torch._thnn import type2backend
import torch
        
def load_model(arch):
    '''
    Args:
        arch: (string) valid torchvision model name,
            recommendations 'vgg16' | 'googlenet' | 'resnet50'
    '''
    if arch == 'softplus50':
        from resnet import resnet50
        model = resnet50()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load('checkpoint.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])        
    else:
        model = models.__dict__[arch](pretrained=True)
    model.cuda()        
    model.eval()
    return model


def cuda_var(tensor, requires_grad=False):
    return Variable(tensor.cuda(), requires_grad=requires_grad)


def upsample(inp, size):
    '''
    Args:
        inp: (Tensor) input
        size: (Tuple [int, int]) height x width
    '''
    backend = type2backend[type(inp)]
    f = getattr(backend, 'SpatialUpSamplingBilinear_updateOutput')
    upsample_inp = inp.new()
    f(backend.library_state, inp, upsample_inp, size[0], size[1])
    return upsample_inp
