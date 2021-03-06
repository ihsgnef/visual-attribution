import torchvision.transforms as transforms
import numpy as np
import torch


class PatternPreprocess(object):
    # only work for VGG16
    def __init__(self, scale_size):
        self.scale = transforms.Compose([
            transforms.Resize(scale_size),
        ])
        self.offset = np.array([103.939, 116.779, 123.68])[:, np.newaxis, np.newaxis]

    def __call__(self, raw_img):
        scaled_img = self.scale(raw_img)
        ret = np.array(scaled_img, dtype=np.float)
        # Channels first.
        ret = ret.transpose(2, 0, 1)
        # To BGR.
        ret = ret[::-1, :, :]
        # Remove pixel-wise mean.
        ret -= self.offset
        ret = np.ascontiguousarray(ret)
        ret = torch.from_numpy(ret).float()

        return ret


def get_preprocess(arch, method, dataset='imagenet'):
    if dataset == "imagenet":
        if arch == 'googlenet':
            transf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[123. / 255, 117. / 255, 104. / 255],
                                     std=[1. / 255, 1. / 255, 1. / 255])
            ])
        elif arch == 'vgg16':
            if method.find('pattern') != -1:  # pattern_net, pattern_lrp
                transf = transforms.Compose([
                    PatternPreprocess((224, 224))
                ])
            else:
                transf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
        elif arch == 'resnet50' or arch == 'resnet18' or 'softplus50':
            if method == 'real_time_saliency':
                normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])
            else:
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

            transf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                #normalize #####################################################
            ])
    if dataset == "cifar10":
        if arch == 'resnet50':    
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),                
            
            transf = transforms.Compose([                
            transforms.ToTensor(),
            #normalize #####################################################
            ])

    return transf



def get_normalize_preprocess(arch, method, dataset='imagenet'):
    if dataset == "imagenet":                                    
        transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),            
        ])
        normalize_transf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        ])

    if dataset == "cifar10":
        transf = transforms.Compose([                
            transforms.ToTensor(),            
        ])
        normalize_transf = transforms.Compose([                
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),                
        ])                
            
    return transf, normalize_transf


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255])
