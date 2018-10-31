import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
from decimal import Decimal
from scipy.stats import spearmanr
from collections import defaultdict

import torch
import torchvision.transforms as transforms

import viz
import utils
from explainers import CASO, \
    VanillaGrad, IntegrateGrad, SmoothGrad, \
    BatchTuner, SmoothCASO, EigenCASO, \
    NewExplainer, VATExplainer
from attackers import EmptyAttack, GhorbaniAttack, ScaledNoiseAttack
from imagenet1000_clsid_to_human import clsid_to_human

import matplotlib.pyplot as plt
import matplotlib.patches as patches


transf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

# norm_tranf = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])


def setup_imagenet(batch_size=16, example_ids=None,
                   n_batches=-1, n_examples=-1,
                   shuffle=True, dump_filenames=None,
                   arch='resnet50'):
    model = utils.load_model(arch)
    model.eval()
    model.cuda()
    print('model loaded')

    image_path = '/fs/imageNet/imagenet/ILSVRC_val/**/*.JPEG'
    image_files = list(glob.iglob(image_path, recursive=True))
    image_files = sorted(image_files, key=lambda x: os.path.basename(x))
    real_ids = [os.path.basename(x) for x in image_files]

    label_path = '/fs/imageNet/imagenet/ILSVRC2012_devkit_t12/' \
                 + 'data/ILSVRC2012_validation_ground_truth.txt'
    with open(label_path) as f:
        labels = [clsid_to_human[int(x)-1] for x in f.readlines()]

    if example_ids is not None:
        examples = {r: (r, m, l)
                    for r, m, l in zip(real_ids, image_files, labels)}
        examples = [examples[x] for x in example_ids]
    else:
        examples = list(zip(real_ids, image_files, labels))

    if shuffle:
        np.random.seed(0)
        np.random.shuffle(examples)

    if n_examples > 0:
        examples = examples[:n_examples]
    elif n_batches > 0:
        examples = examples[:batch_size * n_batches]

    selected_files = sorted([x[0] for x in examples])
    if dump_filenames is not None:
        with open(dump_filenames, 'w') as f:
            f.write(json.dumps(selected_files))
    # print('\n'.join(selected_files))

    def batch_loader(batch):
        batch = list(map(list, zip(*batch)))
        ids, xs, ys = batch
        return (ids, [viz.pil_loader(x) for x in xs], ys)

    batch_indices = list(range(0, len(examples), batch_size))
    batches = [examples[i: i + batch_size] for i in batch_indices]
    batches = map(batch_loader, batches)
    n_batches = len(batch_indices)
    print(n_batches, 'batches', n_batches * batch_size, 'images loaded')
    return model, batches, n_batches


def saliency_correlation(s1, s2):
    # s1 and s2 are batched
    assert s1.shape == s2.shape
    assert s1.ndim == 4
    batch_size = s1.shape[0]
    s1 = viz.agg_clip(s1)
    s2 = viz.agg_clip(s2)
    s1 = s1.reshape(batch_size, -1)
    s2 = s2.reshape(batch_size, -1)
    return [spearmanr(x1, x2).correlation for x1, x2 in zip(s1, s2)]


def saliency_overlap(s1, s2):
    assert s1.shape == s2.shape
    batch_size = s1.shape[0]
    s1 = viz.agg_clip(s1)
    s2 = viz.agg_clip(s2)
    s1 = s1.reshape(batch_size, -1)
    s2 = s2.reshape(batch_size, -1)
    scores = []
    K = 1000
    for x1, x2 in zip(s1, s2):
        x1 = set(np.argsort(-x1)[:K])
        x2 = set(np.argsort(-x2)[:K])
        scores.append(len(x1.intersection(x2)) / K)
    return scores


def get_prediction(model, batch, to_human=True):
    ys = model(batch).max(1)[1].data
    if to_human:
        return [clsid_to_human[y] for y in ys.tolist()]
    else:
        return ys


def plot_matrix(matrix, filename, fontsize=40, rects=[]):
    '''Each entry in the matrix should be a dictionary of:
        image: an image ready to be plotted by imshow
        cmap: color map or None
        text_top, text_bottom, text_left, text_right
        rects: rectangles around box (i, j)
    '''
    n_rows = len(matrix)
    n_cols = max(len(row) for row in matrix)
    plt.rc('text', usetex=True)
    f, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    for i, row in enumerate(matrix):
        for j, c in enumerate(row):
            image = c.get('image', np.zeros((244, 244)))
            cmap = c.get('cmap', None)
            aax = ax[i, j] if len(matrix) > 1 else ax[j]
            aax.imshow(image, cmap=cmap)
            if 'text_top' in c:
                aax.text(0.5, 1.1, c['text_top'], ha='center', va='center',
                         transform=aax.transAxes, fontsize=fontsize)
            if 'text_bottom' in c:
                aax.text(0.5, -0.1, c['text_bottom'], ha='center', va='center',
                         transform=aax.transAxes, fontsize=fontsize)
            if 'text_left' in c:
                aax.text(-0.1, 0.5, c['text_left'], ha='center', va='center',
                         rotation=90, transform=aax.transAxes,
                         fontsize=fontsize)
            if 'text_right' in c:
                aax.text(1.1, 0.5, c['text_right'], ha='center', va='center',
                         rotation=270, transform=aax.transAxes,
                         fontsize=fontsize)
            aax.set_axis_off()
    for i, j in rects:
        aax = ax[i, j] if len(matrix) > 1 else ax[j]
        f.patches.extend([
            patches.Rectangle(
                (-0.01, -0.01), 1.02, 1.02, linewidth=6,
                edgecolor='#55e400', fill=False,
                transform=aax.transAxes)
        ])
    # f.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    f.savefig(filename)
    plt.close('all')


def explain(model, batches, explainers):
    '''
    Returns a dictionary keyed by the id
        image: original input image
        label: predicted label
    The for each (explainer, attacker):
        perturbed: perturbed input
        saliency: saliency with perturbation
    '''
    results = defaultdict(dict)
    for batch_idx, batch in enumerate(batches):
        ids, images, labels = batch
        xs = torch.stack([transf(x) for x in images]).cuda()
        ys = get_prediction(model, xs)
        for explain_name, explainer in explainers:
            saliency_1 = explainer.explain(model, xs.clone()).cpu().numpy()
            for i, id in enumerate(ids):
                results[id]['image'] = images[i]
                results[id]['label'] = labels[i]
                results[id]['prediction'] = ys[i]
                results[id][explain_name] = {'saliency_1': saliency_1[i]}
    return results


def explain_attack_explain(model, batches, explainers, attackers):
    '''
    Returns a dictionary keyed by the id
        image: original input image
        label: predicted label
    for each explainer:
        saliency_1: saliency without perturbation
        for each (explainer, attacker):
            perturbed: perturbed input
            saliency_2: saliency with perturbation
            overlap: between saliency_1 and saliency_2
    '''
    results = defaultdict(dict)
    for batch_idx, batch in enumerate(batches):
        ids, images, labels = batch
        xs = torch.stack([transf(x) for x in images]).cuda()
        ys = get_prediction(model, xs)
        for explain_name, explainer in explainers:
            saliency_1 = explainer.explain(model, xs.clone()).cpu().numpy()
            for attack_name, attacker in attackers:
                perturbed = attacker.attack(model, xs.clone(), saliency_1)
                perturbed_np = perturbed.cpu().numpy()
                saliency_2 = explainer.explain(model, perturbed).cpu().numpy()
                overlap = saliency_overlap(saliency_1, saliency_2)
                for i, id in enumerate(ids):
                    results[id]['image'] = images[i]
                    results[id]['label'] = labels[i]
                    results[id]['prediction'] = ys[i]
                    results[id][explain_name] = {'saliency_1': saliency_1[i]}
                    results[id][(explain_name, attack_name)] = {
                        'perturbed': perturbed_np[i],
                        'saliency_2': saliency_2[i],
                        'overlap': overlap[i],
                    }
    return results


def plot_explainer(model, batches, n_batches, explainers,
                   folder='figures/explain'):
    os.makedirs(folder, exist_ok=True)
    results = explain(model, batches, explainers)
    for id, example in results.items():
        image = transforms.Resize((224, 224))(example['image'])
        row = [{'image': image}]
        for explain_name, _ in explainers:
            saliency_1 = viz.agg_clip(example[explain_name]['saliency_1'])
            row.append({
                'image': saliency_1,
                'cmap': 'gray',
                'text_top': explain_name,
            })
        plot_matrix([row], f'{folder}/{id}.pdf', fontsize=15)


def plot_explainer_attacker(model, batches, n_batches, attackers, explainers,
                            folder='figures/explain_attack'):
    '''For each example, generate a matrix of plots:
    rows are different inputs (original and perturbed)
    columns are different explainers.
    '''
    os.makedirs(folder, exist_ok=True)
    results = explain_attack_explain(model, batches, explainers, attackers)
    for id, example in results.items():
        row = []
        for explain_name, _ in explainers:
            saliency_1 = viz.agg_clip(example[explain_name]['saliency_1'])
            row.append({
                'image': saliency_1,
                'cmap': 'gray',
                'text_top': explain_name,
            })
        matrix = [row]
        for attack_name, _ in attackers:
            image_row = []
            saliency_row = []
            for explain_name, _ in explainers:
                cell = example[(explain_name, attack_name)]
                perturbed = viz.img_rev(cell['perturbed'])
                image_row.append({
                    'image': perturbed,
                    'text_left': attack_name,
                })
                saliency_2 = viz.agg_clip(cell['saliency_2'])
                med_diff = viz.get_median_difference(saliency_2)
                text_top = '{:.3f}'.format(med_diff)
                saliency_row.append({
                    'image': saliency_2,
                    'cmap': 'gray',
                    'text_left': 'Saliency',
                    'text_top': text_top,
                })
            matrix.append(image_row)
            matrix.append(saliency_row)
            plot_matrix(matrix, f'{folder}/{id}.pdf', fontsize=15)


def plot_post_processing(model, batches, n_batches,
                         folder='figures/post_processing'):
    '''Single image saliency mapping with four different post-processing
        methods.
    '''
    os.makedirs(folder, exist_ok=True)
    explainers = [
        ('CASO', CASO()),
        ('CAFO', CASO(lambda_t2=0)),
        ('Gradient', VanillaGrad()),
    ]
    results = explain(model, batches, explainers)
    for id, example in results.items():
        raw_image = transforms.Resize((224, 224))(example['image'])
        image_input = transf(example['image']).numpy()
        plt.rc('text', usetex=True)
        col0 = [
            {'image': raw_image, 'text_left': r'$\Delta$'},
            {'image': raw_image, 'text_left': r'clip$(\Delta)$'},
            {'image': raw_image, 'text_left': r'$\Delta\odot x$'},
            {'image': raw_image, 'text_left': r'clip$(\Delta\odot x)$'},
        ]
        col0 += [{'image': raw_image} for _ in range(3)]
        matrix = [col0]
        for mth_name, _ in explainers:
            col = []
            saliency = example[mth_name]['saliency_1']
            saliency_0 = viz.agg_default(saliency)
            saliency_1 = viz.agg_clip(saliency)
            saliency_2 = viz.agg_default(saliency * image_input)
            saliency_3 = viz.agg_clip(saliency * image_input)
            col.append({'image': saliency_0, 'cmap': 'gray',
                        'text_top': mth_name})
            col.append({'image': saliency_1, 'cmap': 'gray'})
            col.append({'image': saliency_2, 'cmap': 'gray'})
            col.append({'image': saliency_3, 'cmap': 'gray'})
            matrix.append(col)
        matrix = list(map(list, zip(*matrix)))
        plot_matrix(matrix, f'{folder}/{id}.pdf')


def to_decimal(x):
    x = '{:.0e}'.format(Decimal(x))
    x = x.replace('e+', 'e^{')
    x = x.replace('e-', 'e^{-') + '}'
    return x


def plot_steps(model, batches, n_batches, n_iters=None,
               folder='figures/steps'):
    os.makedirs(folder, exist_ok=True)
    if n_iters is None:
        n_iters = [1, 2, 3, 4, 5, 6]
    explainers = []
    for n in n_iters:
        explainers.append(
            (n, CASO(lambda_t2=0, n_iter=n)))
    results = explain(model, batches, explainers)
    for id, example in results.items():
        image = transforms.Resize((224, 224))(example['image'])
        row = [{'image': image}]
        for n in n_iters:
            # only show explainer label on top of the row of original
            # example without perturbation
            cell = example[n]
            saliency = viz.agg_clip(cell['saliency_1'])
            med_diff = viz.get_median_difference(saliency)
            text_top = ''
            row.append({
                'image': saliency,
                'cmap': 'gray',
                'text_top': text_top,
                'text_bottom': r'$\eta={:.3f}$'.format(med_diff),
            })
        path = f'{folder}/{id}.pdf'
        plot_matrix([row], path, fontsize=30)


def plot_l1_l2(model, batches, n_batches, l1s=None, l2s=None,
               folder='figures/l1_l2'):
    os.makedirs(folder, exist_ok=True)
    if l1s is None:
        l1s = [1, 10, 50, 100, 200]
    if l2s is None:
        l2s = [1e2, 1e3, 1e4, 1e5, 1e6]
    explainers = []
    for l1 in l1s:
        # use the combination as name
        for l2 in l2s:
            explainers.append(
                ((l1, l2), EigenCASO(lambda_l1=l1, lambda_l2=l2, init='eig')))
    results = explain(model, batches, explainers)
    rect = (0, 0)
    best_median_diff = 0
    for id, example in results.items():
        matrix = []
        image = transforms.Resize((224, 224))(example['image'])
        for i, l2 in enumerate(l2s):
            text_left = r'$\lambda_2={}$'.format(to_decimal(l2))
            row = [{'image': image, 'text_left': text_left}]
            for j, l1 in enumerate(l1s):
                # only show explainer label on top of the row of original
                # example without perturbation
                cell = example[(l1, l2)]
                saliency = viz.agg_clip(cell['saliency_1'])
                med_diff = viz.get_median_difference(saliency)
                text_top = ''
                if i == 0:
                    text_top = r'$\lambda_1={}$'.format(to_decimal(l1))
                row.append({
                    'image': saliency,
                    'cmap': 'gray',
                    'text_top': text_top,
                    'text_bottom': r'$\eta={:.3f}$'.format(med_diff),
                })
                if med_diff > best_median_diff:
                    rect = (i, j + 1)
                    best_median_diff = med_diff
            matrix.append(row)
        path = f'{folder}/{id}.pdf'
        plot_matrix(matrix, path, fontsize=30, rects=[rect])


def task_goose():
    goose_id = 'ILSVRC2012_val_00045520.JPEG'
    model, batches, n_batches = setup_imagenet(example_ids=[goose_id])
    plot_post_processing(model, batches, n_batches,
                         folder='figures/goose/post_processing')
    plot_l1_l2(model, batches, n_batches, folder='figures/goose/l1_l2')


def task_explain():
    model, batches, n_batches = setup_imagenet(batch_size=1, n_examples=40)
    explainers = [
        ('Grad', VanillaGrad()),
        ('EigenCASO', EigenCASO(lambda_l1=0, n_iter=60)),
    ]
    plot_explainer(model, batches, n_batches, explainers)


def task_explain_attack():
    model, batches, n_batches = setup_imagenet(batch_size=16, n_examples=5)
    attackers = [
        ('Original', EmptyAttack()),
        ('Ghorbani', GhorbaniAttack()),
        ('Random', ScaledNoiseAttack()),
    ]
    explainers = [
        ('Gradient', VanillaGrad()),
        ('SmoothGrad', SmoothGrad()),
        ('IntegratedGrad', IntegrateGrad()),
    ]
    plot_explainer_attacker(model, batches, n_batches, attackers, explainers)


def task_l1_l2():
    model, batches, n_batches = setup_imagenet(n_examples=16)
    n_steps = 16
    l1_lo, l1_hi = 0.01, 2e5
    l2_lo, l2_hi = 1e2, 1e8
    l1s = np.geomspace(l1_lo, l1_hi, n_steps).tolist()
    l2s = np.geomspace(l2_lo, l2_hi, n_steps).tolist()
    plot_l1_l2(model, batches, n_batches, l1s=l1s, l2s=[10])


def task_steps():
    model, batches, n_batches = setup_imagenet(n_examples=16)
    plot_steps(model, batches, n_batches)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    args = parser.parse_args()
    fs = {
        'goose': task_goose,
        'explain': task_explain,
        'explain_attack': task_explain_attack,
        'l1l2': task_l1_l2,
        'steps': task_steps,
    }
    fs[args.task]()
