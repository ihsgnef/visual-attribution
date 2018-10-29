import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict

import torch
import torchvision.transforms as transforms

import viz
import utils
from explainers import CASO, \
    VanillaGrad, IntegrateGrad, SmoothGrad, \
    BatchTuner, SmoothCASO, Eigenvalue, \
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
                   shuffle=True, dump_name=None,
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
    else:
        print('using all images')

    selected_files = sorted([x[0] for x in examples])
    if dump_name is not None:
        with open(dump_name, 'w') as f:
            f.write(json.dumps(selected_files))
    print('\n'.join(selected_files))

    def batch_loader(batch):
        batch = list(map(list, zip(*batch)))
        ids, xs, ys = batch
        return (ids, [viz.pil_loader(x) for x in xs], ys)

    batch_indices = list(range(0, len(examples), batch_size))
    batches = [examples[i: i + batch_size] for i in batch_indices]
    batches = map(batch_loader, batches)
    print('image loaded', len(batch_indices))
    n_batches = len(batch_indices)
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


def plot_explainer_attacker(model, batches, attackers, explainers):
    '''For each example, generate a matrix of plots:
    rows are different inputs (original and perturbed)
    columns are different explainers.
    '''
    os.makedirs('figures/explain_attack', exist_ok=True)
    matrix = []
    results = explain_attack_explain(model, batches, explainers, attackers)
    results = sorted(results.items(), key=lambda x: x[0])
    for id, example in results:
        row = []
        for explain_name, _ in explainers:
            saliency_1 = viz.agg_clip(example[explain_name]['saliency_1'])
            row.append({
                'image': saliency_1,
                'cmap': 'gray',
                'text_top': explain_name,
            })
        matrix.append(row)
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
            plot_matrix(matrix, 'figures/explain_attack/{}.pdf'.format(id),
                        fontsize=15)
            matrix = []


def task_l1_l2(n_examples=3):
    '''Visualize CASO with different l1 and l2'''
    os.makedirs('figures/l1_l2', exist_ok=True)
    model, batches, n_batches = setup_imagenet(batch_size=10,
                                               n_examples=n_examples)
    n_steps = 16
    l1_lo, l1_hi = 0.01, 2e5
    l2_lo, l2_hi = 1e2, 1e8
    l1s = np.geomspace(l1_lo, l1_hi, n_steps).tolist()
    l2s = np.geomspace(l2_lo, l2_hi, n_steps).tolist()

    explainers = []
    for l1 in l1s:
        for l2 in l2s:
            # use the combination as name
            explainers.append(
                ((l1, l2), CASO(lambda_t2=0, lambda_l1=l1, lambda_l2=l2)))

    matrix = []
    resize = transforms.Resize((224, 224))
    results = explain(model, batches, explainers)
    results = sorted(results.items(), key=lambda x: x[0])
    for id, example in results:
        image = resize(example['image'])
        for i, l2 in enumerate(l2s):
            row = [{
                'image': image,
                'text_left': 'l2={:.3f}'.format(l2),
            }]
            for l1 in l1s:
                saliency = viz.agg_clip(example[(l1, l2)]['saliency_2'])
                med_diff = viz.get_median_difference(saliency)
                text_top = 'l1={}\n'.format(l1) if i == 0 else ''
                text_top += '{:.3f}'.format(med_diff)
                row.append({
                    'image': saliency,
                    'cmap': 'gray',
                    'text_top': text_top,
                })
            matrix.append(row)
    plot_matrix(matrix, 'figures/l1_l2/{}.pdf'.format(id))


def plot_histogram_l1(n_examples=4, agg_func=viz.agg_clip):
    l1s = [0, 0.1, 0.5, 1, 10, 100]
    explainers = []
    for l1 in l1s:
        explainers.append((l1, CASO(lambda_l1=l1)))
    model, batches, n_batches = setup_imagenet(n_examples=n_examples)
    results, ids, images, labels = get_saliency_maps(
        model, batches, explainers)
    df_saliency, df_l1, df_idx = [], [], []
    for idx in ids:
        for l1 in l1s:
            s = agg_func(results[idx][l1]['saliency_1'])
            s = s.ravel().tolist()
            df_saliency += s
            df_l1 += [l1 for _ in range(len(s))]
            df_idx += [idx for _ in range(len(s))]

    df = pd.DataFrame({'saliency': df_saliency,
                       'l1': df_l1,
                       'example': df_idx})

    with open('histogram_l1.pkl', 'wb') as f:
        pickle.dump(df, f)


def plot_post_processing(model, batches, example_id):
    '''Single image saliency mapping with four different post-processing
        methods.
    '''
    explainers = [
        ('CASO', BatchTuner(CASO, n_steps=12)),
        ('CAFO', BatchTuner(CASO, lambda_t2=0, n_steps=12)),
        # ('SmoothCAFO', SmoothCASO(lambda_t2=0, n_steps=12)),
        ('Gradient', VanillaGrad()),
        # ('SmoothGrad', SmoothGrad()),
        # ('CASO-E', Eigenvalue()),
        # ('IntegratedGrad', IntegrateGrad()),
    ]
    results, ids, images, labels = get_saliency_maps(
        model, batches, explainers)

    results = results[example_id]
    image_input = transf(images[example_id]).numpy()
    raw_image = transforms.Resize((224, 224))(images[example_id])
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
        saliency = results[mth_name]['saliency_1']
        saliency_0 = viz.agg_default(saliency)
        saliency_1 = viz.agg_clip(saliency)
        saliency_2 = viz.agg_default(saliency * image_input)
        saliency_3 = viz.agg_clip(saliency * image_input)
        col.append({'image': saliency_0, 'cmap': 'gray', 'text_top': mth_name})
        col.append({'image': saliency_1, 'cmap': 'gray'})
        col.append({'image': saliency_2, 'cmap': 'gray'})
        col.append({'image': saliency_3, 'cmap': 'gray'})
        matrix.append(col)
    matrix = list(map(list, zip(*matrix)))
    plot_matrix(matrix, 'figures/goose_1_{}.pdf'.format(example_id))
    print('done', example_id)


def plot_goose_2_full(model, batches, goose_id):
    l1s = [1, 10, 50, 100, 200]
    l2s = [1e2, 1e3, 1e4, 1e5, 1e6]
    l2_tex = ['10^2', '10^3', '10^4', '10^5', '10^6']
    explainers = []
    for l1 in l1s:
        # use the combination as name
        for l2 in l2s:
            explainers.append(
                ((l1, l2), CASO(
                    lambda_l1=l1, lambda_l2=l2)))
    results, ids, images, labels = get_saliency_maps(
        model, batches, explainers)
    matrix = []
    image = transforms.Resize((224, 224))(images[goose_id])
    for i, l2 in enumerate(l2s):
        row = [{'image': image,
                'text_left': r'$\lambda_2={}$'.format(l2_tex[i])}]
        for l1 in l1s:
            # only show explainer label on top of the row of original
            # example without perturbation
            cell = results[goose_id][(l1, l2)]
            saliency = viz.agg_clip(cell['saliency_1'])
            med_diff = viz.get_median_difference(saliency)
            title = r'$\lambda_1={}$'.format(l1) if i == 0 else ''
            row.append({
                'image': saliency,
                'cmap': 'gray',
                'text_top': title,
                'text_bottom': r'$\eta={:.3f}$'.format(med_diff),
            })
        matrix.append(row)
    plot_matrix(matrix, 'figures/goose_2_full.pdf', fontsize=30,
                rects=[(2, 4)])


def plot_goose_2(model, batches, goose_id):
    l1s = [1, 10, 50, 100, 200]
    l2 = 1e4
    explainers = []
    for l1 in l1s:
        explainers.append((l1, CASO(
            lambda_l1=l1, lambda_l2=l2)))
    results, ids, images, labels = get_saliency_maps(
        model, batches, explainers)
    resize = transforms.Resize((224, 224))
    image = resize(images[goose_id])
    row = [{
        'image': image,
        'text_left': r'$\lambda_2=10^4$',
    }]
    for l1 in l1s:
        cell = results[goose_id][l1]
        saliency = viz.agg_clip(cell['saliency_1'])
        med_diff = viz.get_median_difference(saliency)
        row.append({
            'image': saliency,
            'cmap': 'gray',
            'text_top': r'$\lambda_1={}$'.format(l1),
            'text_bottom': r'$\eta={:.3f}$'.format(med_diff),
        })
    matrix = [row]
    plot_matrix(matrix, 'figures/goose_2.pdf', fontsize=30,
                rects=[(1, 4)])


def plot_goose():
    goose_id = 'ILSVRC2012_val_00045520.JPEG'
    model, batches, n_batches = setup_imagenet(example_ids=[goose_id])
    batches = list(batches)
    plot_post_processing(model, batches, goose_id)
    plot_goose_2(model, batches, goose_id)
    plot_goose_2_full(model, batches, goose_id)


def plot_single(model, batches, example_id):
    attackers = [
        ('Original', EmptyAttack()),
        # ('Random', ScaledNoiseAttack()),
        # ('Ghorbani', GhorbaniAttack()),
    ]

    explainers = [
        ('CASO', BatchTuner(CASO, lambda_l2=100)),
        # ('CAFO', BatchTuner(CASO, lambda_t2=0, n_steps=12)),
        # ('SmoothCAFO', SmoothCASO(lambda_t2=0, n_steps=12)),
        # ('Eigen', Eigenvalue()),
        # ('Gradient', VanillaGrad()),
        # ('SmoothGrad', SmoothGrad()),
        # ('CASO-E', Eigenvalue()),
        # ('IntegratedGrad', IntegrateGrad()),
        # ('Eigen', Eigenvalue()),
        # ('New', NewExplainer()),
    ]
    results, ids, images, labels = get_attack_saliency_maps(
        model, batches, explainers, attackers)
    results = results[example_id]
    # construct the matrix to be plotted
    matrix = []
    label = "``{}''".format(labels[example_id])
    for i, (attacker, _) in enumerate(attackers):
        cell = results[(attacker, explainers[0][0])]
        row = [{
                'image': cell['perturbed'],
                'text_top': label if i == 0 else '',
                'text_left': attacker,
        }]
        for explainer, _ in explainers:
            cell = results[(attacker, explainer)]
            s1 = viz.agg_clip(cell['saliency_1'])
            s2 = viz.agg_clip(cell['saliency_2'])
            s = s1 if i == 0 else s2
            med_diff = viz.get_median_difference(s)
            row.append({
                'image': s,
                'cmap': 'gray',
                'text_top': explainer if i == 0 else '',
                'text_bottom': r'{:.3f}'.format(med_diff)
            })
        matrix.append(row)
    plot_matrix(matrix, 'figures/single_{}.pdf'.format(example_id))
    print('done', example_id)


def plot_cherry_pick():
    with open('ghorbani.json') as f:
        example_ids = json.load(f)
    example_ids = example_ids[40:100]
    # goose_id = 'ILSVRC2012_val_00045520.JPEG'
    # example_ids = [goose_id]
    model, batches, n_batches = setup_imagenet(batch_size=1,
                                               example_ids=example_ids)
    batches = list(batches)
    for i, batch in enumerate(batches):
        eid = batch[0][0]
        print(i, eid)
        plot_single(model, [batch], eid)


def task_explain_attack(n_examples=5):
    model, batches, n_batches = setup_imagenet(batch_size=16,
                                               n_examples=n_examples)
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
    plot_explainer_attacker(model, batches, attackers, explainers)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    args = parser.parse_args()
    fs = {
        'explain_attack': task_explain_attack,
        'l1l2': task_l1_l2,
    }
    fs[args.task]()
