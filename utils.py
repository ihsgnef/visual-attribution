from torchvision import models
from torch.autograd import Variable
from torch._thnn import type2backend
import torch
import glob
import numpy as np
import viz
import torchvision
import torchvision.transforms as transforms


def load_model(arch):
    '''
    Args:
        arch: (string) valid torchvision model name,
            recommendations 'vgg16' | 'googlenet' | 'resnet50'
    '''
    print(arch)
    if arch == 'softplus50':
        from resnet import resnet50
        model = resnet50()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load('checkpoint.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
    elif arch == 'cifar50':
        from cifar.models import cifar_resnet
        model = cifar_resnet.cifar_ResNet50()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load('./cifar/checkpoint/ckpt.t7')
        torch.backends.cudnn.enabled = False
        model.load_state_dict(checkpoint['net'])

    else:
        model = models.__dict__[arch](pretrained=True)
    model.cuda()
    model.eval()
    return model


def load_data(batch_size, num_images, transf, dataset='imagenet'):
    batches = []
    if dataset == 'imagenet':
        image_path = '/fs/imageNet/imagenet/ILSVRC_val/**/*.JPEG'
        image_files = list(glob.iglob(image_path, recursive=True))
        np.random.seed(0)
        np.random.shuffle(image_files)
        image_files = image_files[:num_images]
        indices = list(range(0, len(image_files), batch_size))
        for batch_idx, start in enumerate(indices):
            batch = image_files[start: start + batch_size]
            raw_images = [transf(viz.pil_loader(x)) for x in batch]
            raw_images = torch.stack(raw_images)
            batches.append(raw_images)
        return batches

    if dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./cifar/data', train=False, download=True, transform=transforms.ToTensor())#transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        for idx, (inputs, targets) in enumerate(testloader):
            if idx > 0:
                continue   ################################## REMOVE ME and replace with something involving num_images
            batches.append(inputs)
        return batches

    exit("Invalid dataset")


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
