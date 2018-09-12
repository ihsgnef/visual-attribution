import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


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


def VisualizeImageGrayscale(image_3d, percentile=99):
    """Returns a 3D tensor as a grayscale 2D tensor.  This method sums a 3D
    tensor across the absolute value of axis=2, and then clips values at a
    given percentile.
    """
    image_3d = np.abs(image_3d.squeeze())
    image_2d = torch.sum(image_3d, dim=0)

    image_2d = image_2d.numpy()
    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)
    return torch.from_numpy(np.clip((image_2d - vmin) / (vmax - vmin), 0, 1))
