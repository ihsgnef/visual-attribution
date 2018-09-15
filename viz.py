import torch
import numpy as np
from torch.autograd import Variable
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


def VisualizeImageGrayscale(images, percentile=99):
    batch_size, n_chns, height, width = images.shape
    if isinstance(images, Variable):
        images = images.data
    images = images.cpu().numpy()
    images = np.abs(images).sum(axis=1)
    new_images = []
    for i in range(batch_size):
        img = images[i].copy()
        vmax = np.percentile(img, percentile)
        vmin = np.min(img)
        img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
        new_images.append(img)
    new_images = np.stack(new_images)
    assert new_images.shape == (batch_size, height, width)
    return torch.from_numpy(new_images)
