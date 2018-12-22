import pickle
from tasks import plot_matrix, to_decimal

import torchvision.transforms as transforms

from viz import agg_clip


def figure_2(example1, example2):
    '''figure 2 compares CASO with CAFO on two examples, one with high
    confidence and one with low confidence.
    
    each example should be a dictionary with the following entries
    - image (PIL.Image.Image): the original image
    - cafo (numpy.array): shape is (3, 224, 224)
    - caso (numpy.array): shape is (3, 224, 224)
    '''
    image1 = transforms.Resize((224, 224))(example1['image'])
    image2 = transforms.Resize((224, 224))(example2['image'])
    cafo1 = agg_clip(example1['cafo'])
    cafo2 = agg_clip(example2['cafo'])
    caso1 = agg_clip(example1['caso'])
    caso2 = agg_clip(example2['caso'])
    rows = [
        [{'image': image1}, {'image': image2}],
        [
            {'image': cafo1, 'cmap': 'gray', 'text_top': 'CAFO'},
            {'image': cafo2, 'cmap': 'gray', 'text_top': 'CAFO'},
        ],
        [
            {'image': caso1, 'cmap': 'gray', 'text_top': 'CASO'},
            {'image': caso2, 'cmap': 'gray', 'text_top': 'CASO'},
        ],
    ]
    plot_matrix(rows, 'figure_2.pdf', fontsize=15)


def figure_3(examples, lambda1s):
    '''figure 3 compares grad, cafo, and cafo with different sparsity.
    each example should be a dictionary with all three heatmaps, using certain
    lambda. each example should be a dictionary with the following entries:
    - grad
    - cafo
    - caso
    '''
    image = transforms.Resize((224, 224))(examples[0]['image'])
    rows = [[{'image': image}], [{'image': image}], [{'image': image}]]
    for example, lambda1 in zip(examples, lambda1s):
        rows[0].append({'image': agg_clip(example['grad']), 'cmap': 'gray'})
        rows[1].append({'image': agg_clip(example['cafo']), 'cmap': 'gray'})
        rows[2].append({'image': agg_clip(example['caso']), 'cmap': 'gray',
                        'text_bottom': lambda1})
    plot_matrix(rows, 'figure_3.pdf', fontsize=15)


def figure_4(regular_example, smooth_example):
    '''figure 4 compares non-smooth and smooth version of gradient, integrated
    gradient, cafo, and caso. each example should be a dictionary with the
    following entries:
    - grad
    - integrated
    - cafo
    - caso
    '''
    image = transforms.Resize((224, 224))(regular_example['image'])
    regular_row = [
        {'image': image, 'text_left': 'Regular'},
        {'image': agg_clip(regular_example['grad']), 'cmap': 'gray', 'text_top': 'Grad'},
        {'image': agg_clip(regular_example['integrated']), 'cmap': 'gray', 'text_top': 'IG'},
        {'image': agg_clip(regular_example['cafo']), 'cmap': 'gray', 'text_top': 'CAFO'},
        {'image': agg_clip(regular_example['caso']), 'cmap': 'gray', 'text_top': 'caso'},
    ]
    smooth_row = [
        {'image': image, 'text_left': 'Smooth'},
        {'image': agg_clip(smooth_example['grad']), 'cmap': 'gray'},
        {'image': agg_clip(smooth_example['integrated']), 'cmap': 'gray'},
        {'image': agg_clip(smooth_example['cafo']), 'cmap': 'gray'},
        {'image': agg_clip(smooth_example['caso']), 'cmap': 'gray'},
    ]
    plot_matrix([regular_row, smooth_row], 'figure_4.pdf', fontsize=15)


if __name__ == '__main__':
    placeholder = pickle.load(open('placeholder_0.pkl', 'rb'))
    figure_2(placeholder, placeholder)
    figure_3([placeholder for _ in range(6)], list(range(6)))
    figure_4(placeholder, placeholder)
