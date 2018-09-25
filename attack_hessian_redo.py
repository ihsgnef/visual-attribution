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
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

import viz
import utils
from explainers_redo import zero_grad
from explainers_redo import SparseExplainer, RobustSparseExplainer, \
    VanillaGradExplainer, IntegrateGradExplainer, SmoothGradExplainer, \
    LambdaTunerExplainer, BatchTuner

import matplotlib.pyplot as plt
import matplotlib.patches as patches


transf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])


def get_topk_mask(saliency, k=1e4, topk_agg=None, flip=False):
    '''Generate binary mask based on saliency value.
    Saliency is first aggregated via `topk_agg`, and the mask has the same
    shape as the aggregated saliency (3D or 4D).
    If `flip` is False, high saliency value corresponds to 1, otherwise 0.
    :param saliency: 4D numpy.array
    :param k: number of pixels to attack, channel independent if mask is 4D
    :param topk_agg: should probably move to whoever calls me
    :param flip: if flip, high value is 0 instead of 1
    '''
    assert len(saliency.shape) == 4
    batch_size, n_chs, height, width = saliency.shape
    if topk_agg is not None:
        saliency = topk_agg(saliency)

    if len(saliency.shape) == 4:
        saliency = saliency.reshape(batch_size, n_chs, -1)
        topk_mask = np.ones_like(saliency) if flip else np.zeros_like(saliency)
        topk_idx = np.argsort(-saliency, axis=2)[:, :, :k]
        for i in range(batch_size):
            for j in range(n_chs):
                topk_mask[i, j, topk_idx[i, j]] = 0 if flip else 1
        topk_mask = topk_mask.reshape(batch_size, n_chs, height, width)
    elif len(saliency.shape) == 3:
        saliency = saliency.reshape(batch_size, -1)
        topk_mask = np.ones_like(saliency) if flip else np.zeros_like(saliency)
        topk_idx = np.argsort(-saliency, axis=1)[:, :k]
        for i in range(batch_size):
            topk_mask[i, topk_idx[i]] = 0 if flip else 1
        topk_mask = topk_mask.reshape(batch_size, height, width)
        topk_mask = np.expand_dims(topk_mask, axis=1)
        topk_mask = np.tile(topk_mask, (1, n_chs, 1, 1))
    else:
        print('saliency shape wrong')
    return topk_mask


class EmptyAttack:

    def attack(self, model, x, saliency=None):
        return x


class GhorbaniAttack:

    def __init__(self,
                 lambda_t1=0,
                 lambda_t2=1,
                 lambda_l1=0,
                 lambda_l2=0,
                 n_iter=20,
                 optim='sgd',
                 lr=1e-2,
                 epsilon=2/255,
                 k=1e4,
                 topk_agg=lambda x: np.abs(x)
                 ):
        '''
        :param lambda_t1
        :param lambda_t2
        :param lambda_l1
        :param lambda_l2
        :param n_iter
        :param optim
        :param lr
        :param epsilon
        :param k: number of pixels to dampen
        :param topk_agg: aggregation for selecting top-k pixels
        '''
        self.lambda_t1 = lambda_t1
        self.lambda_t2 = lambda_t2
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.n_iter = n_iter
        self.optim = optim.lower()
        self.lr = lr
        self.epsilon = epsilon
        self.k = int(k)
        self.topk_agg = topk_agg

    def get_input_grad(self, x, output, y, create_graph=False):
        '''two methods for getting input gradient'''
        loss = F.cross_entropy(output, y)
        x_grad, = torch.autograd.grad(loss, x, create_graph=create_graph)

        # grad_out = torch.zeros_like(output.data)
        # grad_out.scatter_(1, y.data.unsqueeze(0).t(), 1.0)
        # x_grad, = torch.autograd.grad(output, x,
        #                               grad_outputs=grad_out,
        #                               create_graph=create_graph)

        return x_grad

    def attack(self, model, x, saliency=None):
        '''Generate attack against specified saliency mapping.
        If saliency is not specified, assume vanila gradient saliency.
        '''
        batch_size, n_chs, height, width = x.shape

        x_prev = x.clone()
        delta = torch.zeros_like(x)
        y_org = model(Variable(x)).max(1)[1].data

        if saliency is None:
            x_curr = Variable(x, requires_grad=True)
            output = model(x_curr)
            y = output.max(1)[1]
            saliency = self.get_input_grad(x_curr, output, y)
            saliency = saliency.data.cpu().numpy()

        topk_mask = get_topk_mask(saliency, self.k, topk_agg=self.topk_agg)
        topk_mask = torch.FloatTensor(topk_mask).cuda()
        topk_mask = Variable(topk_mask)

        step_size = self.epsilon / self.n_iter
        stopped = [False for _ in range(batch_size)]
        for i in range(self.n_iter):
            zero_grad(model)
            x_curr = Variable(x, requires_grad=True)
            output = model(x_curr)
            y = output.max(1)[1]

            x_grad = self.get_input_grad(x_curr, output, y, create_graph=True)

            topk = (x_grad.abs() * topk_mask).sum()
            delta, = torch.autograd.grad(-topk, x_curr)
            delta = delta.sign().data

            # verify same prediction
            for idx in range(batch_size):
                if stopped[idx] or y.data[idx] != y_org[idx]:
                    x[idx] = x_prev[idx]
                    delta[idx].zero_()
                    stopped[idx] = True

            if all(stopped):
                break

            x_prev = x.clone()
            x = torch.clamp(x + step_size * delta, 0, 1)

        # final check that all predictions remain
        y = model(Variable(x)).max(1)[1].data
        assert (y == y_org).all()
        return x


class ScaledNoiseAttack:

    def __init__(self, epsilon=2 / 255):
        self.epsilon = epsilon

    def attack(self, model, x, saliency=None):
        x = x.cpu().numpy()
        noise = 2 * np.random.randint(2, size=x.shape) - 1
        noise = np.sign(noise) * self.epsilon
        x = np.clip(x + noise * x, 0, 1)
        x = torch.FloatTensor(x).cuda()
        return x


class FGSM:

    def __init__(self, epsilon=2 / 255, n_iter=10):
        self.epsilon = epsilon
        self.n_iter = n_iter

    def get_input_grad(self, x, output, y, create_graph=False):
        '''two methods for getting input gradient'''
        loss = F.cross_entropy(output, y)
        x_grad, = torch.autograd.grad(loss, x, create_graph=create_graph)

        # grad_out = torch.zeros_like(output.data)
        # grad_out.scatter_(1, y.data.unsqueeze(0).t(), 1.0)
        # x_grad, = torch.autograd.grad(output, x,
        #                               grad_outputs=grad_out,
        #                               create_graph=create_graph)
        return x_grad

    def attack(self, model, x, saliency=None):
        batch_size, n_chs, height, width = x.shape
        step_size = self.epsilon / self.n_iter
        for i in range(self.n_iter):
            zero_grad(model)
            x_curr = Variable(x, requires_grad=True)
            output = model(x_curr)
            y = output.max(1)[1]
            x_grad = self.get_input_grad(x, output, y).data
            x_grad = x_grad.sign()
            x = torch.clamp(x + step_size * x_grad, 0, 1)
        return x


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
                   shuffle=True, dump_name=None):
    model = utils.load_model('resnet50')
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
    # example_ids = example_ids[:n_examples]
    n_examples = len(example_ids)
    model, batches = setup_imagenet(example_ids=example_ids)

    attackers = [
        ('Ghorbani', GhorbaniAttack()),
        ('Random', ScaledNoiseAttack()),
    ]

    explainers = [
        # ('CASO', SparseExplainer()),
        # ('Robust', RobustSparseExplainer()),
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
    df.to_pickle('ghorbani_1000_baselines.pkl')
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
    mth_name, explainer = ('CASO', BatchTuner())
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
    f, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    for i, row in enumerate(matrix):
        for j, c in enumerate(row):
            image = c.get('image', None)
            cmap = c.get('cmap', None)
            title = c.get('title', None)
            yalign = c.get('yalign', 0.5)
            rotate_title = c.get('rotate_title', False)
            xlabel = c.get('xlabel', None)

            aax = ax[i, j] if len(matrix) > 1 else ax[j]
            if image is None:
                aax.imshow(np.zeros((244, 244)), cmap='gray')
                aax.imshow(np.zeros((244, 244)), cmap='gray')
            if cmap is None:
                aax.imshow(image)
            else:
                aax.imshow(image, cmap=cmap)
            if title is not None:
                if rotate_title:
                    aax.set_title(title, rotation=90,
                                  x=-0.1, y=yalign, fontsize=fontsize)
                else:
                    # aax.set_title(c['title'], fontsize=fontsize)
                    aax.text(0.5, 1.1, title,
                             horizontalalignment='center',
                             verticalalignment='center',
                             transform=aax.transAxes,
                             fontsize=fontsize)
            if xlabel is not None:
                aax.text(0.5, -0.1, xlabel,
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=aax.transAxes,
                         fontsize=fontsize)
            aax.set_axis_off()
    for i, j in rects:
        aax = ax[i, j] if len(matrix) > 1 else ax[j]
        f.patches.extend([
            patches.Rectangle(
                (-0.1, -0.1), 1.1, 1.1, linewidth=6, 
                edgecolor='g', fill=False,
                transform=aax.transAxes)
        ])
    f.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
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


def plot_explainer_attacker(n_examples=6, agg_func=viz.agg_clip):
    model, batches = setup_imagenet(n_examples=n_examples)

    attackers = [
        ('Original', EmptyAttack()),  # empty attacker so perturbed = original
        # ('Ghorbani', GhorbaniAttack()),
        # ('Random', ScaledNoiseAttack()),
    ]

    explainers = [
        # ('CASO', SparseExplainer()),
        ('Batch', BatchTuner()),
        # ('CASO', LambdaTunerExplainer()),
        ('Gradient', VanillaGradExplainer()),
        ('SmoothGrad', SmoothGradExplainer()),
        ('IntegratedGrad', IntegrateGradExplainer()),
    ]

    results, ids, images, labels = get_attack_saliency_maps(
        model, batches, explainers, attackers)
    assert len(results) == n_examples

    # construct the matrix to be plotted
    matrix = []
    ids = sorted(ids)  # ordered by real ids
    for idx in ids:
        label = '"{}"'.format(labels[idx])
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
                    'title': title
                })
            matrix.append(row)
    plot_matrix(matrix, 'figures/explainer_attacker.pdf', fontsize=15)


def plot_l1_l2(agg_func=viz.agg_clip):
    example_ids = ['ILSVRC2012_val_00019603.JPEG']
    model, batches = setup_imagenet(example_ids=example_ids)

    attackers = [
        # ('Original', EmptyAttack()),
        ('Ghorbani', GhorbaniAttack()),
    ]

    n_steps = 16
    l1_lo, l1_hi = 0.5, 2000
    l2_lo, l2_hi = 1e2, 1e8
    l1s = np.geomspace(l1_lo, l1_hi, n_steps)
    l2s = np.geomspace(l2_lo, l2_hi, n_steps)

    explainers = []
    for l1 in l1s:
        # use the combination as name
        for l2 in l2s:
            explainers.append(
                ((l1, l2), SparseExplainer(lambda_l1=l1, lambda_l2=l2)))

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
                'title': 'l2={}'.format(l2),
                'rotate_title': True
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
                    'title': title,
                })
            matrix.append(row)
    plot_matrix(matrix, 'figures/l1_l2.pdf')


def plot_histogram_l1(n_examples=4, agg_func=viz.agg_clip):
    l1s = [0, 0.1, 0.5, 1, 10, 100]
    explainers = []
    for l1 in l1s:
        explainers.append((l1, SparseExplainer(lambda_l1=l1)))
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
        ('CASO', SparseExplainer(lambda_l1=100, lambda_l2=1e4)),
        # ('CASO', BatchTuner()),
        ('Gradient', VanillaGradExplainer()),
        ('SmoothGrad', SmoothGradExplainer()),
        ('IntegratedGrad', IntegrateGradExplainer()),
    ]
    results, ids, images, labels = get_saliency_maps(
        model, batches, explainers)

    results = results[goose_id]
    image_input = transf(images[goose_id]).numpy()
    raw_image = transforms.Resize((224, 224))(images[goose_id])

    # 0: delta
    # 1: clip(delta)
    # 2: delta * input
    # 3: clip(delta * input)
    plt.rc('text', usetex=True)

    col0 = [
        {'image': raw_image,
         'title': r'$\Delta$',
         'rotate_title': True},
        {'image': raw_image,
         'title': r'clip$(\Delta)$',
         'rotate_title': True,
         'yalign': 0.6},
        {'image': raw_image,
         'title': r'$\Delta\odot x$',
         'rotate_title': True,
         'yalign': 0.6},
        {'image': raw_image,
         'title': r'clip$(\Delta\odot x)$',
         'rotate_title': True,
         'yalign': 0.75},
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
        # TODO add median difference
        col.append({'image': saliency_0, 'cmap': 'gray', 'title': mth_name})
        col.append({'image': saliency_1, 'cmap': 'gray'})
        col.append({'image': saliency_2, 'cmap': 'gray'})
        col.append({'image': saliency_3, 'cmap': 'gray'})
        matrix.append(col)
    matrix = list(map(list, zip(*matrix)))
    plot_matrix(matrix, 'figures/goose_1.pdf')


def plot_goose_2_full(model, batches, goose_id):
    l1s = [1, 10, 50, 100, 200]
    l2s = [1e2, 1e3, 1e4, 1e5, 1e6]
    l2_tex = ['10^2', '10^3', '10^4', '10^5', '10^6']

    explainers = []
    for l1 in l1s:
        # use the combination as name
        for l2 in l2s:
            explainers.append(
                ((l1, l2), SparseExplainer(
                    lambda_l1=l1, lambda_l2=l2)))

    results, ids, images, labels = get_saliency_maps(
        model, batches, explainers)
    matrix = []
    image = transforms.Resize((224, 224))(images[goose_id])
    for i, l2 in enumerate(l2s):
        row = [{
            'image': image,
            'title': r'$\lambda_2={}$'.format(l2_tex[i]),
            'rotate_title': True,
            'yalign': 0.65
        }]
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
                'title': title,
                'xlabel': r'$\eta={:.3f}$'.format(med_diff),
            })
        matrix.append(row)
    plot_matrix(matrix, 'figures/goose_2_full.pdf', fontsize=30,
                rects=[(2, 4)])


def plot_goose_2(model, batches, goose_id):
    l1s = [1, 10, 50, 100, 200]
    l2 = 1e4

    explainers = []
    for l1 in l1s:
        explainers.append((l1, SparseExplainer(
            lambda_l1=l1, lambda_l2=l2)))

    results, ids, images, labels = get_saliency_maps(
        model, batches, explainers)
    resize = transforms.Resize((224, 224))
    image = resize(images[goose_id])
    row = [{
        'image': image,
        'title': r'$\lambda_2=10^4$',
        'rotate_title': True,
        'yalign': 0.65
    }]
    for l1 in l1s:
        cell = results[goose_id][l1]
        saliency = viz.agg_default(cell['saliency_1'])
        med_diff = viz.get_median_difference(saliency)
        row.append({
            'image': saliency,
            'cmap': 'gray',
            'title': r'$\lambda_1={}$'.format(l1),
            'xlabel': r'$\eta={:.3f}$'.format(med_diff),
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
    }
    fs[args.task]()
