import torch
import numpy as np
from torch.autograd import Variable
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pylab as P

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def plot_cam(attr, xi, cmap='jet', alpha=0.5):
    attr -= attr.min()
    attr /= (attr.max() + 1e-20)

    plt.imshow(xi)
    plt.imshow(attr, alpha=alpha, cmap=cmap)


def plot_bbox(bboxes, xi, linewidth=1):
    ax = plt.gca()
    ax.imshow(xi)

    if not isinstance(bboxes[0], list):
        bboxes = [bboxes]

    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                 linewidth=linewidth, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')


def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
        P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)


def VisualizeImageGrayscale(imgs, percentile=99):
    batch_size, n_chns, height, width = imgs.shape
    if isinstance(imgs, Variable):
        imgs = imgs.data
    imgs = torch.abs(imgs).sum(dim=1)
    imgs = imgs.view(batch_size, -1)
    imgs_cpu = imgs.numpy()
    vmax = np.percentile(imgs_cpu, percentile, axis=1)
    vmax = torch.FloatTensor(vmax).unsqueeze(1)
    vmin = torch.min(imgs, dim=1)[0].unsqueeze(1)
    imgs = torch.clamp((imgs - vmin) / (vmax - vmin), 0, 1)
    imgs = imgs.view(batch_size, height, width)
    return imgs
