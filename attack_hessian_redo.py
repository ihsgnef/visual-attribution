import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from collections import defaultdict

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

import viz
import utils
from explainers import CASO, RobustCASO, \
    VanillaGradExplainer, IntegrateGradExplainer, SmoothGradExplainer, \
    LambdaTunerExplainer, BatchTuner, SmoothCASO, Eigenvalue
from attackers import EmptyAttack, GhorbaniAttack, ScaledNoiseAttack

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
    from imagenet1000_clsid_to_human import clsid_to_human
    ys = model(Variable(batch)).max(1)[1].data
    if to_human:
        return [clsid_to_human[y] for y in ys]
    else:
        return ys


def attack_batch(model, batch, explainers, attackers,
                 return_saliency=False):
    results = []
    batch_size = batch.shape[0]
    for mth_name, explainer in explainers:
        saliency_1 = explainer.explain(model, batch.clone()).cpu().numpy()
        for atk_name, attacker in attackers:
            perturbed = attacker.attack(model, batch.clone(), saliency_1)
            perturbed_np = perturbed.cpu().numpy()
            saliency_2 = explainer.explain(model, perturbed).cpu().numpy()
            # print(atk_name, mth_name, saliency_1.shape, saliency_2.shape)
            scores = saliency_overlap(saliency_1, saliency_2)

            for i in range(batch_size):
                row = {
                    'idx': i,
                    'explainer': mth_name,
                    'attacker': atk_name,
                    'overlap': scores[i],
                }
                if return_saliency:
                    row.update({
                        'saliency_1': saliency_1[i],
                        'saliency_2': saliency_2[i],
                        'perturbed': perturbed_np[i],
                    })
                results.append(row)
    return results


def setup_imagenet(batch_size=16, example_ids=None,
                   n_batches=-1, n_examples=-1,
                   shuffle=True, dump_name=None,
                   arch='softplus50'):
    model = utils.load_model(arch)
    model.eval()
    model.cuda()
    print('model loaded')

    image_path = '/fs/imageNet/imagenet/ILSVRC_val/**/*.JPEG'
    image_files = list(glob.iglob(image_path, recursive=True))
    image_files = sorted(image_files, key=lambda x: os.path.basename(x))
    real_ids = [os.path.basename(x) for x in image_files]

    from imagenet1000_clsid_to_human import clsid_to_human

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
    return model, batches


def run_attack_baselines(n_examples=4):
    with open('ghorbani.json') as f:
        example_ids = json.load(f)
    example_ids = example_ids[:10]
    n_examples = len(example_ids)
    model, batches = setup_imagenet(batch_size=12, example_ids=example_ids)

    attackers = [
        ('Ghorbani', GhorbaniAttack()),
        ('Random', ScaledNoiseAttack()),
    ]

    explainers = [
        ('CASO', BatchTuner(CASO, n_steps=12)),
        ('CASO-1', BatchTuner(CASO, n_steps=12, t2_lo=0, t2_hi=0)),
        ('CASO-R', BatchTuner(RobustCASO, n_steps=12)),
        ('Gradient', VanillaGradExplainer()),
        ('SmoothGrad', SmoothGradExplainer()),
        ('IntegratedGrad', IntegrateGradExplainer()),
    ]
    results = []
    n_batches = int(n_examples / 16)
    for batch_idx, batch in enumerate(tqdm(batches, total=n_batches)):
        ids, xs, ys = batch
        xs = torch.stack([transf(x) for x in xs]).cuda()
        rows = attack_batch(model, xs, explainers, attackers)
        for i, row in enumerate(rows):
            rows[i]['idx'] = ids[row['idx']]
        results += rows
    df = pd.DataFrame(results)
    # df.to_pickle('ghorbani_1000_baselines.pkl')
    df.drop(['idx'], axis=1)
    print(df.groupby(['attacker', 'explainer']).mean())


def run_attack_tuner(n_examples=4):
    with open('ghorbani.json') as f:
        example_ids = json.load(f)
    # example_ids = example_ids[:n_examples]
    n_examples = len(example_ids)
    model, batches = setup_imagenet(example_ids=example_ids)
    attackers = [
        ('Ghorbani', GhorbaniAttack()),
        ('Random', ScaledNoiseAttack()),
    ]
    mth_name, explainer = ('CASO', BatchTuner(n_steps=12))
    results = []
    n_batches = int(n_examples / 16)
    for batch_idx, batch in enumerate(tqdm(batches, total=n_batches)):
        ids, xs, ys = batch
        xs = torch.stack([transf(x) for x in xs]).cuda()
        batch = xs
        rows = []
        batch_size = batch.shape[0]
        saliency_1, l11, l21 = explainer.explain(model, batch.clone(),
                                                 get_lambdas=True)
        saliency_1 = saliency_1.cpu().numpy()
        for atk_name, attacker in attackers:
            perturbed = attacker.attack(model, batch.clone(), saliency_1)
            perturbed_np = perturbed.cpu().numpy()
            saliency_2, l12, l22 = explainer.explain(model, perturbed,
                                                     get_lambdas=True)
            saliency_2 = saliency_2.cpu().numpy()
            scores = saliency_overlap(saliency_1, saliency_2)
            for i in range(batch_size):
                row = {
                    'idx': i,
                    'explainer': mth_name,
                    'attacker': atk_name,
                    'overlap': scores[i],
                    'perturbed': perturbed_np,
                    'l11': l11[i],
                    'l21': l21[i],
                    'l12': l12[i],
                    'l22': l22[i],
                }
                rows.append(row)
        for i, row in enumerate(rows):
            rows[i]['idx'] = ids[row['idx']]
        results += rows
    df = pd.DataFrame(results)
    df.to_pickle('ghorbani_1000_tuner.pkl')
    df.drop(['idx'], axis=1)
    print(df.groupby(['attacker', 'explainer']).mean())


def plot_matrix(matrix, filename, fontsize=40, rects=[]):
    '''Each entry in the matrix should be a dictionary of:
    image: an image ready to be plotted by imshow
    cmap: color map or None
    title: title
    rotate_title: if True show on the left of the image
    '''
    n_rows = len(matrix)
    n_cols = len(matrix[0])
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


def get_saliency_maps(model, batches, explainers):
    results = defaultdict(dict)
    all_images = dict()
    all_labels = dict()
    all_ids = []
    for batch_idx, batch in enumerate(batches):
        ids, images, labels = batch
        xs = torch.stack([transf(x) for x in images]).cuda()
        ys = get_prediction(model, xs)
        for i, (idx, img, lab) in enumerate(zip(ids, images, labels)):
            all_images[idx] = img
            all_labels[idx] = ys[i]
            all_ids.append(idx)
        for mth_name, explainer in explainers:
            saliency_1 = explainer.explain(model, xs.clone()).cpu().numpy()
            if hasattr(explainer, 'history'):
                history = explainer.history
            else:
                history = None
            for i, s1 in enumerate(saliency_1):
                idx = ids[i]
                row = {
                    'idx': idx,
                    'explainer': mth_name,
                    'saliency_1': s1,
                    'history': history,
                }
                results[idx][mth_name] = row
    return results, all_ids, all_images, all_labels


def get_attack_saliency_maps(model, batches, explainers, attackers):
    '''Collect saliency mappings.
    Returns a dictionary keyed by the example ids, each entry has:
    idx: example id (real id)
    attacker: name of the attacker
    explainer: name of the explainer
    perturbed: perturbed input
    saliency_1: saliency without perturbation
    saliency_2: saliency with perturbation
    '''
    results = defaultdict(dict)
    all_images = dict()
    all_labels = dict()
    all_ids = []
    for batch_idx, batch in enumerate(batches):
        ids, images, labels = batch

        xs = torch.stack([transf(x) for x in images]).cuda()
        ys = get_prediction(model, xs)
        rows = attack_batch(model, xs, explainers, attackers,
                            return_saliency=True)
        for i, (idx, img, lab) in enumerate(zip(ids, images, labels)):
            all_images[idx] = img
            all_labels[idx] = ys[i]
            all_ids.append(idx)
        for i, row in enumerate(rows):
            row['idx'] = ids[row['idx']]
            attacker = row['attacker']
            explainer = row['explainer']
            ptb = np.array(row['perturbed']).swapaxes(0, 2).swapaxes(0, 1)
            row['perturbed'] = np.uint8(ptb * 255)
            results[row['idx']][(attacker, explainer)] = row
    return results, all_ids, all_images, all_labels


def plot_explainer_attacker(n_examples=3, agg_func=viz.agg_clip):
    model, batches = setup_imagenet(batch_size=16, n_examples=n_examples,
                                    arch='softplus50')

    attackers = [
        ('Original', EmptyAttack()),  # empty attacker so perturbed = original
        # ('Ghorbani', GhorbaniAttack()),
        ('Random', ScaledNoiseAttack()),
    ]

    explainers = [
        # ('CASO-VAT', BatchTuner(CASO, n_steps=12, init='vat')),
        # ('CASO', BatchTuner(CASO, n_steps=12)),
        # ('CASO-1', BatchTuner(CASO, n_steps=12, t2_lo=0, t2_hi=0)),
        # ('CASO-2', BatchTuner(CASO, n_steps=12, lambda_t1=0)),
        ('SmoothCASO', SmoothCASO(n_steps=12)),
        # ('CASO-R', BatchTuner(RobustCASO, n_steps=12)),
        ('Gradient', VanillaGradExplainer()),
        ('SmoothGrad', SmoothGradExplainer()),
        # ('IntegratedGrad', IntegrateGradExplainer()),
    ]

    results, ids, images, labels = get_attack_saliency_maps(
        model, batches, explainers, attackers)
    assert len(results) == n_examples
    df = []

    # construct the matrix to be plotted
    matrix = []
    ids = sorted(ids)  # ordered by real ids
    for idx in ids:
        label = "``{}''".format(labels[idx])
        for i, (attacker, _) in enumerate(attackers):
            cell = results[idx][(attacker, explainers[0][0])]
            # show label on top if empty attacker
            # otherwise show attack name on the left
            row = [{
                    'image': cell['perturbed'],
                    'title': label if i == 0 else attacker,
                    'rotate_title': False if i == 0 else True,
            }]
            for explainer, _ in explainers:
                # only show explainer label on top of the row of original
                # example without perturbation
                cell = results[idx][(attacker, explainer)]
                s1 = agg_func(cell['saliency_1'])
                s2 = agg_func(cell['saliency_2'])
                s = s1 if i == 0 else s2
                med_diff = viz.get_median_difference(s)
                title = explainer + '\n' if i == 0 else ''
                title += '{:.3f}'.format(med_diff)
                row.append({
                    'image': s,
                    'cmap': 'gray',
                    'text_top': title
                })
                df.append({
                    'attacker': attacker,
                    'explainer': explainer,
                    'overlap': cell['overlap'],
                })
            matrix.append(row)
    plot_matrix(matrix, 'figures/explainer_attacker.pdf', fontsize=15)
    df = pd.DataFrame(df)
    print(df.groupby(['attacker', 'explainer']).mean())


def plot_l1_l2(agg_func=viz.agg_clip):
    with open('ghorbani.json') as f:
        example_ids = json.load(f)
    example_id = 3
    example_ids = [example_ids[example_id]]
    model, batches = setup_imagenet(batch_size=10, example_ids=example_ids)

    attackers = [
        ('Original', EmptyAttack()),
        # ('Ghorbani', GhorbaniAttack()),
    ]

    n_steps = 16
    l1_lo, l1_hi = 0.01, 2e5
    l2_lo, l2_hi = 1e2, 1e8
    l1s = np.geomspace(l1_lo, l1_hi, n_steps)
    l2s = np.geomspace(l2_lo, l2_hi, n_steps)

    explainers = []
    for l1 in l1s:
        # use the combination as name
        for l2 in l2s:
            explainers.append(
                ((l1, l2), CASO(lambda_t2=0, lambda_l1=l1, lambda_l2=l2)))

    results, ids, images, labels = get_attack_saliency_maps(
        model, batches, explainers, attackers)
    # results, ids, images, labels = get_saliency_maps(
    #     model, batches, explainers)
    resize = transforms.Resize((224, 224))
    # construct the matrix to be plotted
    matrix = []
    ids = sorted(ids)  # ordered by real ids
    for idx in ids:
        attacker = attackers[0][0]
        image = resize(images[idx])
        for i, l2 in enumerate(l2s):
            row = [{
                'image': image,
                'text_left': 'l2={:.3f}'.format(l2),
            }]
            for l1 in l1s:
                # only show explainer label on top of the row of original
                # example without perturbation
                cell = results[idx][(attacker, (l1, l2))]
                # cell = results[idx][(l1, l2)]
                saliency = agg_func(cell['saliency_2'])
                med_diff = viz.get_median_difference(saliency)
                title = 'l1={}\n'.format(l1) if i == 0 else ''
                title += '{:.3f}'.format(med_diff)
                row.append({
                    'image': saliency,
                    'cmap': 'gray',
                    'text_top': title,
                })
            matrix.append(row)
    plot_matrix(matrix, 'figures/l1_l2_{}.pdf'.format(example_id))


def plot_histogram_l1(n_examples=4, agg_func=viz.agg_clip):
    l1s = [0, 0.1, 0.5, 1, 10, 100]
    explainers = []
    for l1 in l1s:
        explainers.append((l1, CASO(lambda_l1=l1)))
    model, batches = setup_imagenet(n_examples=n_examples)
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


def plot_goose_1(model, batches, goose_id):
    explainers = [
        ('CASO', BatchTuner(CASO, n_steps=12)),
        ('CAFO', BatchTuner(CASO, lambda_t2=0, n_steps=12)),
        # ('CASOR', BatchTuner(RobustCASO, n_steps=12)),
        # ('SmoothCAFO', SmoothCASO(lambda_t2=0, n_steps=12)),
        ('Gradient', VanillaGradExplainer()),
        # ('SmoothGrad', SmoothGradExplainer()),
        # ('CASO-E', Eigenvalue()),
        # ('IntegratedGrad', IntegrateGradExplainer()),
    ]
    results, ids, images, labels = get_saliency_maps(
        model, batches, explainers)

    results = results[goose_id]
    image_input = transf(images[goose_id]).numpy()
    raw_image = transforms.Resize((224, 224))(images[goose_id])
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
    plot_matrix(matrix, 'figures/goose_1_{}.pdf'.format(goose_id))
    print('done', goose_id)


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
    model, batches = setup_imagenet(example_ids=[goose_id])
    batches = list(batches)
    plot_goose_1(model, batches, goose_id)
    plot_goose_2(model, batches, goose_id)
    plot_goose_2_full(model, batches, goose_id)


def plot_single(model, batches, example_id):
    attackers = [
        ('Original', EmptyAttack()),
        ('Random', ScaledNoiseAttack()),
        ('Ghorbani', GhorbaniAttack()),
    ]

    explainers = [
        ('CASO', BatchTuner(CASO, n_steps=12)),
        ('CAFO', BatchTuner(CASO, lambda_t2=0, n_steps=12)),
        # ('CASOR', BatchTuner(RobustCASO, n_steps=12)),
        # ('SmoothCAFO', SmoothCASO(lambda_t2=0, n_steps=12)),
        ('Gradient', VanillaGradExplainer()),
        # ('SmoothGrad', SmoothGradExplainer()),
        # ('CASO-E', Eigenvalue()),
        # ('IntegratedGrad', IntegrateGradExplainer()),
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
                'text_bottom': 'r{:.3f}'.format(med_diff)
            })
        matrix.append(row)
    plot_matrix(matrix, 'figures/single_{}.pdf'.format(example_id))
    print('done', example_id)


def plot_cherry_pick():
    with open('ghorbani.json') as f:
        example_ids = json.load(f)
    example_ids = example_ids[20:100]
    # goose_id = 'ILSVRC2012_val_00045520.JPEG'
    # example_ids = [goose_id]
    model, batches = setup_imagenet(batch_size=1, example_ids=example_ids)
    batches = list(batches)
    for i, batch in enumerate(batches):
        eid = batch[0][0]
        print(i, eid)
        plot_single(model, [batch], eid)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    args = parser.parse_args()
    fs = {
        'baselines': run_attack_baselines,
        'tuner': run_attack_tuner,
        'l1l2': plot_l1_l2,
        'attacks': plot_explainer_attacker,
        'histogram': plot_histogram_l1,
        'goose': plot_goose,
        'cherry': plot_cherry_pick,
    }
    fs[args.task]()
