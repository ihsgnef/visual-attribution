import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import viz
import utils
from create_explainer import get_explainer
from preprocess import get_preprocess

import matplotlib.pyplot as plt
from PIL import Image
from plotnine import ggplot, aes, geom_density, facet_grid


def zero_grad(x):
    if isinstance(x, Variable):
        if x.grad is not None:
            x.grad.data.zero_()
    elif isinstance(x, torch.nn.Module):
        for p in x.parameters():
                if p.grad is not None:
                    p.grad.data.zero_()


class NoiseAttack:

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def attack_with_saliency(self, inp, saliency):
        return self.attack(inp)

    def attack(self, inp):
        noise = 2 * np.random.randint(2, size=inp.shape) - 1
        noise = np.sign(noise) * self.epsilon
        perturbed = np.clip(inp.cpu().numpy() + noise, 0, 1)
        perturbed = torch.FloatTensor(perturbed)
        noise = torch.FloatTensor(noise)
        return perturbed, noise


class ScaledNoiseAttack:

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def attack_with_saliency(self, inp, saliency):
        return self.attack(inp)

    def attack(self, inp):
        inp = inp.cpu().numpy()
        noise = 2 * np.random.randint(2, size=inp.shape) - 1
        noise = np.sign(noise) * self.epsilon
        perturbed = np.clip(inp + noise * inp, 0, 1)
        perturbed = torch.FloatTensor(perturbed)
        noise = torch.FloatTensor(noise)
        return perturbed, noise


class FGSM:

    def __init__(self, model, epsilon=2 / 255, n_iterations=10):
        self.model = model
        self.epsilon = epsilon
        self.n_iterations = n_iterations

    def attack_with_saliency(self, inp, saliency):
        return self.attack(inp, saliency)

    def attack(self, inp, saliency=None):
        batch_size, n_chs, height, width = inp.shape

        inp_org = inp.clone()
        batch_size, n_chs, height, width = inp.shape
        step_size = self.epsilon / self.n_iterations
        for i in range(self.n_iterations):
            zero_grad(model)
            new_inp = Variable(inp, requires_grad=True)
            output = self.model(new_inp)
            out_loss = F.cross_entropy(output, output.max(1)[1])
            inp_grad, = torch.autograd.grad(
                out_loss, new_inp, create_graph=True)
            inp_grad = inp_grad.view(batch_size, n_chs, -1)
            delta = inp_grad.sign().data
            inp = torch.clamp(inp + step_size * delta, 0, 1)
        return inp, inp - inp_org


def get_topk_mask(saliency, k=1e4, topk_agg=None, flip=False):
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


class GhorbaniAttack:

    def __init__(self, model,
                 lambda_t1=1, lambda_t2=1,
                 lambda_l1=1e4, lambda_l2=1e4,
                 n_iterations=10, optim='sgd', lr=1e-2,
                 epsilon=2 / 255, const_k=1e4, topk_agg=None):
        self.model = model
        self.lambda_t1 = lambda_t1
        self.lambda_t2 = lambda_t2
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.n_iterations = n_iterations
        self.optim = optim.lower()
        self.lr = lr
        self.epsilon = epsilon
        self.const_k = int(const_k)
        self.topk_agg = topk_agg

    def attack_with_saliency(self, inp, saliency):
        return self.attack(inp, saliency)

    def _backprop(self, inp, ind):
        zero_grad(self.model)
        output = self.model(inp)
        # if ind is None:
        ind = output.data.max(1)[1]
        grad_out = output.data.clone()
        grad_out = torch.zeros_like(output.data)
        grad_out.scatter_(1, ind.unsqueeze(0).t(), 1.0)
        # output.backward(grad_out, create_graph=True)
        inp_grad, = torch.autograd.grad(output, inp, grad_outputs=grad_out,
                                        create_graph=True)
        return inp.grad

    def attack(self, inp, saliency=None):
        inp_org = inp.clone()
        batch_size, n_chs, height, width = inp.shape

        prev = inp.clone()
        delta = torch.zeros_like(inp)
        ind_org = self.model(Variable(inp)).max(1)[1].data

        if saliency is None:
            new_inp = Variable(inp, requires_grad=True)
            output = self.model(new_inp)
            out_loss = F.cross_entropy(output, output.max(1)[1])
            saliency, = torch.autograd.grad(out_loss, new_inp)
            saliency = saliency.data

        topk_mask = get_topk_mask(saliency.cpu().numpy(),
                                  self.const_k,
                                  topk_agg=self.topk_agg)
        topk_mask = torch.FloatTensor(topk_mask).cuda()
        topk_mask = Variable(topk_mask)

        step_size = self.epsilon / self.n_iterations
        stopped = [False for _ in range(batch_size)]
        for i in range(self.n_iterations):
            zero_grad(model)
            new_inp = Variable(inp, requires_grad=True)
            output = self.model(new_inp)
            ind = output.max(1)[1]

            '''two methods for getting input gradient'''
            # inp_grad = self._backprop(new_inp, ind)
            out_loss = F.cross_entropy(output, ind)
            inp_grad, = torch.autograd.grad(out_loss, new_inp,
                                            create_graph=True)

            topk = (inp_grad.abs() * topk_mask).sum()
            delta, = torch.autograd.grad(-topk, new_inp)
            delta = delta.sign().data

            # verify same prediction
            for bidx in range(batch_size):
                if stopped[bidx] or ind.data[bidx] != ind_org[bidx]:
                    inp[bidx] = prev[bidx]
                    delta[bidx].zero_()
                    stopped[bidx] = True

            if all(stopped):
                break

            prev = inp.clone()
            inp = torch.clamp(inp + step_size * delta, 0, 1)

        # final check that all predictions remain
        ind_new = self.model(Variable(inp)).max(1)[1].data
        assert (ind_org == ind_new).all()
        return inp, inp - inp_org


def saliency_correlation(s1, s2, image):
    # s1 and s2 are batched
    s1 = aggregate(s1, image)
    s2 = aggregate(s2, image)
    assert s1.shape == s2.shape
    assert s1.ndimension() == 3  # batch, height, width
    batch_size = s1.shape[0]
    s1 = s1.cpu().numpy().reshape(batch_size, -1)
    s2 = s2.cpu().numpy().reshape(batch_size, -1)
    scores = []
    for x1, x2 in zip(s1, s2):
        scores.append(spearmanr(x1, x2).correlation)
    return scores


def channel_correlation(s1, s2, image):
    assert s1.shape == s2.shape
    assert s1.ndimension() == 4  # batch, 3, height, width
    s1 = s1.abs()
    s2 = s2.abs()
    batch_size = s1.shape[0]
    s1 = s1.cpu().numpy().reshape(batch_size, 3, -1)
    s2 = s2.cpu().numpy().reshape(batch_size, 3, -1)
    scores = []
    for x1, x2 in zip(s1, s2):
        scores.append(spearmanr(x1[k], x2[k]).correlation for k in range(3))
    scores = list(map(list, zip(*scores)))
    return scores


def saliency_overlap(s1, s2, image):
    assert s1.shape == s2.shape
    batch_size = s1.shape[0]
    s1 = s1.cpu().numpy().reshape(batch_size, -1)
    s2 = s2.cpu().numpy().reshape(batch_size, -1)
    scores = []
    K = 1000
    for x1, x2 in zip(s1, s2):
        x1 = set(np.argsort(-x1)[:K])
        x2 = set(np.argsort(-x2)[:K])
        scores.append(len(x1.intersection(x2)) / K)
    return scores


def binit(saliency):
    '''normalize saliency values then bin'''
    batch_size, _, height, width = saliency.shape
    s = saliency.cpu().numpy()
    s = np.abs(s).sum(1)
    s = s.reshape(batch_size, -1)
    # vmax = np.expand_dims(np.percentile(s, 90), 1)
    # vmin = np.expand_dims(np.percentile(s, 10), 1)
    vmax = np.expand_dims(np.max(s, 1), 1)
    vmin = np.expand_dims(np.min(s, 1), 1)
    s = (s - vmin) / (vmax - vmin)
    s = np.digitize(s, np.arange(0, 1, 1 / 10)) / 10
    s = np.clip(s, 0, 1)
    s = s.reshape(batch_size, height, width)
    return torch.FloatTensor(s)


def aggregate(saliency, image):
    '''combine saliency mapping with image
    from 4D (bsz, 3, h, w) to 3D (bsz, h, w)
    '''
    saliency = saliency.cpu()
    if image is not None:
        image = image.cpu()
    # return binit(saliency)
    # return viz.VisualizeImageGrayscale(saliency)
    return saliency.abs().sum(dim=1)
    # return (saliency * image).abs().sum(dim=1)
    # return saliency.max(dim=1)[0]
    # return saliency.abs().max(dim=1)[0]
    # return saliency.sum(dim=1)
    # return saliency.sum(dim=1).abs()
    # return (saliency * image).sum(dim=1)
    # return (saliency * image).abs().max(dim=1)[0]
    # return (saliency * image).max(dim=1)[0]


def perturb(image, delta, mask, flip=False):
    '''perturb image with delta within mask
    create zero-one mask where locations with high mask value get one
    if flip is True, flip the sign of zero-one mask
    '''
    K = 10000
    assert len(image.shape)
    assert mask.ndimension() == 3  # batch, height, width
    assert image.shape == delta.shape
    batch_size, n_chs, height, width = image.shape
    image = image.cpu().numpy()
    delta = delta.cpu().numpy()
    mask = mask.cpu().numpy()
    mask = mask.reshape(batch_size, -1)
    zero_mask = np.zeros_like(mask)
    mask = np.argsort(-mask, axis=1)[:, :K]
    for i in range(batch_size):
        zero_mask[i][mask[i]] = 1
    zero_mask = zero_mask.reshape(batch_size, height, width)
    zero_mask = np.expand_dims(zero_mask, 1)
    if flip:
        zero_mask = 1 - zero_mask
    perturbed = image + delta * zero_mask
    return torch.from_numpy(perturbed).cuda()


def saliency_histogram(configs, model, raw_images):
    results = []
    for mth_name, kwargs in configs:
        explainer = get_explainer(model, mth_name, kwargs)
        batch_size = len(raw_images)
        images = torch.stack([transf(x) for x in raw_images]).cuda()
        batch_size, n_chs, height, width = images.shape
        inputs = Variable(images.clone(), requires_grad=True)
        saliency = explainer.explain(inputs).cpu().numpy()
        saliency = saliency.reshape(batch_size, n_chs, height * width)
        # don't aggregate, look at channels separately and in combination
        for i in range(batch_size):
            for j, chn in enumerate(['R', 'G', 'B']):
                for v in saliency[i][j]:
                    results.append([mth_name, chn, v])
            for v in saliency[i].sum(1):
                results.append([mth_name, 'sum', v])
    return results


def attack_with_saliency_test(configs, attackers, model, raw_images,
                              get_saliency_maps=False):
    images = torch.stack([transf(x) for x in raw_images]).cuda()

    '''run saliency methods'''
    results = []
    saliency_maps = []
    batch_size = images.shape[0]
    for mth_name, kwargs in configs:
        explainer = get_explainer(model, mth_name, kwargs)
        inputs = Variable(images.clone().cuda(), requires_grad=True)
        saliency_1 = explainer.explain(inputs)
        map_1 = saliency_1.cpu().numpy()

        for atk_name, atk in attackers:
            perturbed, delta = atk.attack_with_saliency(images.clone(),
                                                        saliency_1.clone())
            ptb_np = perturbed.cpu().numpy()
            # unrestricted perturbation
            inputs = Variable(perturbed.cuda(), requires_grad=True)
            saliency_2 = explainer.explain(inputs)
            map_2 = saliency_2.cpu().numpy()

            scores = [
                saliency_correlation(saliency_1, saliency_2, inputs.data),
                *channel_correlation(saliency_1, saliency_2, inputs.data),
            ]
            scores = list(map(list, zip(*scores)))

            for i in range(batch_size):
                results.append([mth_name, atk_name] + scores[i])
                if get_saliency_maps:
                    saliency_maps.append([i, mth_name, atk_name,
                                          map_1[i], map_2[i], ptb_np[i]])

    if get_saliency_maps:
        return results, saliency_maps
    else:
        return results


def attack_test(configs, attackers, model, raw_images,
                get_saliency_maps=False):
    '''construct attacks'''
    attacks = []
    images = torch.stack([transf(x) for x in raw_images]).cuda()
    for atk_name, atk in attackers:
        perturbed, delta = atk.attack(images.clone())
        ptb_np = perturbed.cpu().numpy()
        attacks.append((atk_name, perturbed, delta, ptb_np))

    '''run saliency methods'''
    results = []
    saliency_maps = []
    batch_size = images.shape[0]
    for mth_name, kwargs in configs:
        explainer = get_explainer(model, mth_name, kwargs)
        inputs = Variable(images.clone().cuda(), requires_grad=True)
        saliency_1 = explainer.explain(inputs)
        map_1 = saliency_1.cpu().numpy()

        for atk_name, perturbed, delta, ptb_np in attacks:
            # unrestricted perturbation
            inputs = perturbed.clone()
            inputs = Variable(inputs.cuda(), requires_grad=True)
            saliency_2 = explainer.explain(inputs)
            map_2 = saliency_2.cpu().numpy()

            # # only perturb highlighted region
            # inputs = perturb(images, delta, saliency_1)
            # inputs = Variable(inputs.cuda(), requires_grad=True)
            # saliency_3 = explainer.explain(inputs)

            # # perturb outside highlighted region
            # inputs = perturb(images, delta, saliency_1, flip=True)
            # inputs = Variable(inputs.cuda(), requires_grad=True)
            # saliency_4 = explainer.explain(inputs)

            scores = [
                saliency_correlation(saliency_1, saliency_2, inputs.data),
                *channel_correlation(saliency_1, saliency_2, inputs.data),
                # saliency_correlation(saliency_1, saliency_3, inputs.data),
                # saliency_correlation(saliency_1, saliency_4, inputs.data),

                # saliency_overlap(saliency_1, saliency_2, inputs.data),
                # saliency_overlap(saliency_1, saliency_3, inputs.data),
                # saliency_overlap(saliency_1, saliency_4, inputs.data),

                # saliency_overlap(saliency_1, delta, inputs.data),
                # saliency_overlap(saliency_2, delta, inputs.data),
                # saliency_overlap(saliency_3, delta, inputs.data),
                # saliency_overlap(saliency_4, delta, inputs.data),
            ]
            scores = list(map(list, zip(*scores)))

            for i in range(batch_size):
                results.append([mth_name, atk_name] + scores[i])
                if get_saliency_maps:
                    saliency_maps.append([i, mth_name, atk_name,
                                          map_1[i], map_2[i], ptb_np[i]])

    if get_saliency_maps:
        return results, saliency_maps
    else:
        return results


def setup_imagenet(batch_size):

    def batch_loader(image_files):
        return [viz.pil_loader(x) for x in image_files]

    image_path = '/fs/imageNet/imagenet/ILSVRC_val/**/*.JPEG'
    image_files = list(glob.iglob(image_path, recursive=True))
    np.random.seed(0)
    np.random.shuffle(image_files)
    batch_indices = list(range(0, len(image_files), batch_size))
    image_files = [image_files[i: i + batch_size] for i in batch_indices]
    image_batches = map(batch_loader, image_files)
    n_batches = len(image_files)
    print('image path loaded', n_batches)

    model = utils.load_model('resnet50')
    model.eval()
    model.cuda()
    print('model loaded')

    transf = get_preprocess('resnet50', 'sparse')
    return image_batches, n_batches, model, transf


def setup_cifar10(batch_size):

    def batch_loader(loaded):
        inputs, targets = loaded
        return inputs

    data = torchvision.datasets.CIFAR10(
        root='./cifar/data', train=False, download=True,
        transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(data, batch_size, shuffle=False)
    n_batches = len(loader)
    batches = map(batch_loader, loader)
    print('image path loaded', n_batches)

    model = utils.load_model('cifar50')
    model.cuda()
    model.eval()
    print('model loaded')

    def null(x):
        return x

    # transf = get_preprocess('resnet50', 'sparse', 'cifar10')
    return batches, n_batches, model, null


image_batches, n_batches, model, transf = setup_imagenet(16)
EPSILON = 2 / 255
GHO_K = int(1e4)
# image_batches, n_batches, model, transf = setup_cifar10(16)
# EPSILON = 16 / 255
# GHO_K = 400
TIMES_INPUT = False


attackers = [
    ('gho',
     GhorbaniAttack(
         model,
         lambda_t1=0,
         lambda_t2=1,
         lambda_l1=0,
         lambda_l2=0,
         n_iterations=30,
         optim='sgd',
         lr=1e-2,
         epsilon=EPSILON,
         const_k=GHO_K,
         topk_agg=lambda x: np.abs(x),
         # topk_agg=lambda x: np.abs(x).sum(1),
     )),
    # ('fgsm', FGSM(model, epsilon=EPSILON, n_iterations=10)),
    # ('rnd', NoiseAttack(epsilon=EPSILON)),
    ('srnd', ScaledNoiseAttack(epsilon=EPSILON)),
]


configs = [
    ('sparse zero',
     {
         'lambda_t1': 1,
         'lambda_t2': 1,
         'lambda_l1': 100,
         'lambda_l2': 1e4,
         'n_iterations': 10,
         'optim': 'sgd',
         'lr': 0.1,
         'times_input': TIMES_INPUT,
         'init': 'zero',
     }),
    ('vanilla_grad', None),
    ('smooth_grad', None),
    ('integrate_grad', None),
]


def run_attack_short(n_batches):
    results = []
    for batch_idx, batch in enumerate(image_batches):
        if batch_idx >= n_batches:
            break
        # scores = attack_test(configs, attackers, model, batch)
        scores = attack_with_saliency_test(configs, attackers, model, batch)
        results += scores
    n_scores = len(results[0]) - 2  # number of different scores
    columns = (
        ['method', 'attack'] +
        ['score_{}'.format(i) for i in range(n_scores)]
    )
    df = pd.DataFrame(results, columns=columns)
    print(df.groupby(['attack', 'method']).mean())

    # read previous results
    # with open('output/results.812.json') as f:
    #     df = df.append(pd.DataFrame(json.load(f)), ignore_index=True)


def run_attack_long():
    results = []

    def check():
        n_scores = len(results[0]) - 2  # number of different scores
        columns = (
            ['method', 'attack'] +
            ['score_{}'.format(i) for i in range(n_scores)]
        )
        df = pd.DataFrame(results, columns=columns)
        with open('output/results.{}.json'.format(batch_idx), 'w') as f:
            f.write(df.to_json())
        df = df.groupby(['attack', 'method']).mean()
        # print(df)

    for batch_idx, batch in enumerate(tqdm(image_batches, total=n_batches)):
        if batch_idx % 20 == 0 and batch_idx > 0:
            check()
            results = []
        results += attack_test(configs, attackers, model, batch)
    if len(results) > 0:
        check()


def run_histogram(n_batches):
    results = []
    for batch_idx, batch in enumerate(tqdm(image_batches, total=n_batches)):
        if batch_idx >= n_batches:
            break
        results += saliency_histogram(configs, model, batch)
    columns = (
        ['method', 'channel', 'saliency']
    )
    df = pd.DataFrame(results, columns=columns)
    p = (
        ggplot(df)
        + aes(x='saliency')
        + geom_density()
        + facet_grid('method ~ channel')
    )
    p.save('histogram.pdf')


def name_conv(name):
    name = name.split(' ')[0]
    conv = {
        'sparse': 'H1',
        'robust_sparse': 'H2',
        'vat': 'VAT',
        'vanilla_grad': 'Gradient',
        'smooth_grad': 'SmoothGrad',
        'integrate_grad': 'IntegratedGrad',
        'gho': 'Ghorbani',
        'srnd': 'Random',
    }
    return conv[name]


def get_saliency_maps(n_images, configs, attackers):
    maps = []
    images = []
    cnt = 0
    for batch_idx, batch in enumerate(image_batches):
        batch = batch[:n_images]  # hacky way of handling small n
        _, _maps = attack_with_saliency_test(configs, attackers, model, batch,
                                             get_saliency_maps=True)
        for i, _ in enumerate(_maps):
            _maps[i][0] += len(maps)
        maps += _maps
        images += batch
        cnt += len(batch)
        if cnt >= n_images:
            break
    return maps, images


def figures(n_images):
    maps, images = get_saliency_maps(n_images, configs, attackers)
    all_saliency_maps = dict()
    image_labels = dict()
    for batch_idx, mth, atk, map1, map2, ptb in maps:
        map1 = viz.VisualizeImageGrayscale(
            torch.FloatTensor(np.array(map1)).unsqueeze(0))
        map1 = map1.squeeze(0).numpy()

        map2 = viz.VisualizeImageGrayscale(
            torch.FloatTensor(np.array(map2)).unsqueeze(0))
        map2 = map2.squeeze(0).numpy()

        ptb = np.array(ptb).swapaxes(0, 2).swapaxes(0, 1)
        ptb = np.uint8(ptb * 255)
        # TODO
        if batch_idx not in all_saliency_maps:
            all_saliency_maps[batch_idx] = dict()
        atk = name_conv(atk)
        mth = name_conv(mth)
        all_saliency_maps[batch_idx][(atk, mth)] = {
            'label': 'label', 'pred1': 'pred1', 'pred2': 'pred2',
            'map1': map1, 'map2': map2, 'ptb': ptb}
        image_labels[batch_idx] = 'label'

    methods = [name_conv(x[0]) for x in configs]
    attacks = [name_conv(x[0]) for x in attackers]

    title_fontsize = 20
    resize = transforms.Resize((224, 224))

    '''figure 2'''
    rows = n_images
    cols = len(methods) + 1
    print('figure 2 size: {} rows, {} cols'.format(rows, cols))
    f, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for image_idx, image in enumerate(images):
        image = resize(image)
        label = image_labels[image_idx]
        ax[image_idx, 0].imshow(image)
        ax[image_idx, 0].axis('off')
        ax[image_idx, 0].set_title('Input\n("{}")'.format(label),
                                   fontsize=title_fontsize)
        saliency_maps = all_saliency_maps[image_idx]
        for j, mth in enumerate(methods):
            map1 = saliency_maps[(attacks[0], mth)]['map1']
            ax[image_idx,  j + 1].imshow(map1, cmap='gray')
            ax[image_idx,  j + 1].axis('off')
            ax[image_idx,  j + 1].set_title(mth, fontsize=title_fontsize)
    f.tight_layout()
    f.savefig('figures/figure2_multi.pdf')

    def figure3_mask(image, map1):
        image = np.array(image)
        height, width, _ = image.shape
        map1 = map1.ravel()
        map1 = np.argsort(-map1)[:GHO_K]
        image = image.reshape(-1, 3)
        image[map1, :] = 255
        image = image.reshape(height, width, 3)
        image = Image.fromarray(image)
        return image

    '''figure 3'''
    # TODO
    rows = n_images
    cols = len(methods) + 1
    print('figure 3 size: {} rows, {} cols'.format(rows, cols))
    f, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    ax[0, 0].set_title('input', fontsize=title_fontsize)
    for image_idx, image in enumerate(images):
        image = resize(image)
        label = image_labels[image_idx]
        ax[image_idx, 0].imshow(image)
        ax[image_idx, 0].axis('off')
        ax[image_idx, 0].set_title('Input\n("{}")'.format(label),
                                   fontsize=title_fontsize)
        saliency_maps = all_saliency_maps[image_idx]
        for j, mth in enumerate(methods):
            map1 = saliency_maps[(attacks[0], mth)]['map1']
            map1 = figure3_mask(image, map1)
            pred2 = saliency_maps[(attacks[0], mth)]['pred2']
            ax[image_idx,  j + 1].imshow(map1)
            ax[image_idx,  j + 1].axis('off')
            ax[image_idx,  j + 1].set_title('{}\n("{}")'.format(mth, pred2),
                                            fontsize=title_fontsize)
    f.tight_layout()
    f.savefig('figures/figure3_multi.pdf')

    '''figure 5s'''
    for image_idx, image in enumerate(images):
        rows = len(attacks) + 1
        cols = len(methods) + 1
        print('figure 5[{}] size: {} rows, {} cols'.format(
            image_idx, rows, cols))
        f, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

        image = resize(image)
        label = image_labels[image_idx]
        ax[0, 0].imshow(image)
        ax[0, 0].axis('off')
        ax[0, 0].set_title('Input\n("{}")'.format(label),
                           fontsize=title_fontsize)
        # vertical text to the left of the top left image
        # ax[0, 0].text(x=0, y=0,
        #               s="original",
        #               size=title_fontsize,
        #               rotation='vertical',
        #               horizontalalignment='center',
        #               verticalalignment='center')
        saliency_maps = all_saliency_maps[image_idx]
        for j, mth in enumerate(methods):
            map1 = saliency_maps[(attacks[0], mth)]['map1']
            ax[0, j + 1].imshow(map1, cmap='gray')
            ax[0, j + 1].axis('off')
            ax[0, j + 1].set_title(mth, fontsize=title_fontsize)
        for i, atk in enumerate(attacks):
            ptb = saliency_maps[(atk, mth)]['ptb']
            ax[i + 1, 0].imshow(ptb)
            ax[i + 1, 0].axis('off')
            ax[i + 1, 0].set_title(atk, rotation='vertical', x=-0.1, y=0.5,
                                   fontsize=title_fontsize)
            for j, mth in enumerate(methods):
                map2 = saliency_maps[(atk, mth)]['map2']
                ax[i + 1, j + 1].imshow(map2, cmap='gray')
                ax[i + 1, j + 1].axis('off')
        f.tight_layout()
        f.savefig('figures/figure5_{}.pdf'.format(image_idx))


def figure_config_attack(configs, attackers, filename, n_images):
    '''x-axis is configs, y is attacks'''
    maps, images = get_saliency_maps(n_images, configs, attackers)
    all_saliency_maps = dict()
    image_labels = dict()
    for batch_idx, mth, atk, map1, map2, ptb in maps:
        map1 = viz.VisualizeImageGrayscale(
            torch.FloatTensor(np.array(map1)).unsqueeze(0))
        map1 = map1.squeeze(0).numpy()

        map2 = viz.VisualizeImageGrayscale(
            torch.FloatTensor(np.array(map2)).unsqueeze(0))
        map2 = map2.squeeze(0).numpy()

        ptb = np.array(ptb).swapaxes(0, 2).swapaxes(0, 1)
        ptb = np.uint8(ptb * 255)
        # TODO
        if batch_idx not in all_saliency_maps:
            all_saliency_maps[batch_idx] = dict()
        atk = name_conv(atk)
        all_saliency_maps[batch_idx][(atk, mth)] = {
            'label': 'label', 'pred1': 'pred1', 'pred2': 'pred2',
            'map1': map1, 'map2': map2, 'ptb': ptb}
        image_labels[batch_idx] = 'label'

    methods = [x[0] for x in configs]
    attacks = [name_conv(x[0]) for x in attackers]

    title_fontsize = 20
    resize = transforms.Resize((224, 224))

    rows = (len(attacks) + 1) * n_images
    cols = len(configs) + 1
    print('figure size: {} rows, {} cols'.format(rows, cols))
    f, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    row_idx = 0
    for batch_idx, saliency_maps in all_saliency_maps.items():
        image = resize(images[batch_idx])
        label = image_labels[batch_idx]
        ax[row_idx, 0].imshow(image)
        ax[row_idx, 0].axis('off')
        ax[row_idx, 0].set_title(label, fontsize=title_fontsize)
        for i, atk in enumerate(attacks):
            ptb = saliency_maps[(atk, mth)]['ptb']
            ax[row_idx + i + 1, 0].imshow(ptb)
            ax[row_idx + i + 1, 0].axis('off')
            ax[row_idx + i + 1, 0].set_title(atk, rotation='vertical',
                                             x=-0.1, y=0.5,
                                             fontsize=title_fontsize)
        for j, mth in enumerate(methods):
            map1 = saliency_maps[(attacks[0], mth)]['map1']
            ax[row_idx, j + 1].imshow(map1, cmap='gray')
            ax[row_idx, j + 1].axis('off')
            ax[row_idx, j + 1].set_title(mth, fontsize=title_fontsize)
            for i, atk in enumerate(attacks):
                map2 = saliency_maps[(atk, mth)]['map2']
                ax[row_idx + i + 1, j + 1].imshow(map2, cmap='gray')
                ax[row_idx + i + 1, j + 1].axis('off')
        row_idx += len(attacks) + 1

    f.tight_layout()
    f.savefig(filename)


default_sparse_config = {
        'lambda_t1': 1,
        'lambda_t2': 1,
        'lambda_l1': 100,
        'lambda_l2': 1e4,
        'n_iterations': 10,
        'optim': 'sgd',
        'lr': 0.1,
        'times_input': TIMES_INPUT,
        'init': 'zero',
    }


def figure_l2_attack(n_images):
    configs = []
    for lambda_l2 in [0, 1, 10, 1e2, 1e3, 1e4, 1e5]:
        config = default_sparse_config.copy()
        config.update({'lambda_l2': lambda_l2})
        configs.append(('sparse (l2={})'.format(lambda_l2), config))
    figure_config_attack(configs, attackers,
                         'figures/figure_l2_attack.pdf',
                         n_images)


def figure_l1_attack(n_images):
    configs = []
    for lambda_l1 in [0, 1, 10, 1e2, 1e3, 1e4, 1e5]:
        config = default_sparse_config.copy()
        config.update({'lambda_l1': lambda_l1})
        configs.append(('sparse (l1={})'.format(lambda_l1), config))
    figure_config_attack(configs, attackers,
                         'figures/figure_l1_attack.pdf',
                         n_images)


def figure_niter_attack(n_images):
    configs = []
    for n_iterations in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        config = default_sparse_config.copy()
        config.update({'n_iterations': n_iterations})
        configs.append(('sparse (niter={})'.format(n_iterations), config))
    figure_config_attack(configs, attackers,
                         'figures/figure_niter_attack.pdf',
                         n_images)


if __name__ == '__main__':
    run_attack_short(n_batches=3)
