import glob
import numpy as np
import pytablewriter
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import viz
import utils
from create_explainer import get_explainer
from preprocess import get_preprocess
from param_matrix import get_saliency, get_saliency_no_viz
from resnet import resnet50

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class NoiseAttack(object):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def attack(self, inp):
        inp = inp.data.cpu().numpy()
        noise = 2 * np.random.randint(2, size=inp.shape) - 1
        perturb = np.sign(noise) * self.epsilon
        # return torch.FloatTensor(perturb).cuda()
        fused = np.clip(inp + perturb, 0, 1)
        fused = torch.FloatTensor(fused).cuda().squeeze()
        return fused


class NewHessianAttack(object):

    def __init__(self, model,
                 lambda_t1=1, lambda_t2=1,
                 lambda_l1=1e4, lambda_l2=1e4,
                 n_iterations=10, optim='sgd', lr=1e-2,
                 epsilon=16 / 255):
        self.model = model
        self.lambda_t1 = lambda_t1
        self.lambda_t2 = lambda_t2
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.n_iterations = n_iterations
        self.optim = optim.lower()
        self.lr = lr
        self.epsilon = epsilon

    def attack(self, inp):
        inp_org = inp.data.cpu().numpy()
        ind_org = self.model(inp).max(1)[1].data.cpu().numpy()

        batch_size, n_chs, img_height, img_width = inp.shape
        step_size = self.epsilon / self.n_iterations
        accu_perturb = np.zeros_like(inp_org)
        prev_perturb = np.zeros_like(inp_org)
        for i in range(self.n_iterations):
            inp = np.clip(inp_org + accu_perturb, 0, 1)
            inp = Variable(torch.from_numpy(inp).cuda(), requires_grad=True)

            output = self.model(inp)
            ind = output.max(1)[1]

            if ind.data.cpu().numpy() != ind_org:
                print('stop', i)
                accu_perturb = prev_perturb
                break

            out_loss = F.cross_entropy(output, ind)
            inp_grad, = torch.autograd.grad(out_loss, inp, create_graph=True)
            inp_grad = inp_grad.squeeze().view(n_chs, -1)

            topk = inp_grad.abs().sort(dim=1, descending=True)[0]
            topk = topk[:, :1000].sum()
            delta, = torch.autograd.grad(-topk, inp)
            delta = delta.sign().data.cpu().numpy()
            prev_perturb = accu_perturb
            accu_perturb = accu_perturb + step_size * delta

        fused = np.clip(inp_org + accu_perturb, 0, 1)
        fused = torch.FloatTensor(fused).cuda().squeeze()
        return fused
        # return torch.FloatTensor(accu_perturb).cuda()


def get_prediction(model, inp):
    inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)
    output = model(inp)
    ind = output.max(1)[1]
    return ind


def saliency_correlation(saliency_1, saliency_2):
    saliency_1 = saliency_1.cpu().numpy()  # already normalized maps
    saliency_2 = saliency_2.cpu().numpy()
    saliency_1 = saliency_1.ravel()
    saliency_2 = saliency_2.ravel()
    return spearmanr(saliency_1, saliency_2)


# def fuse(inp, delta, mask=None, gamma=3e-1):
#     '''use saliency as a mask and fuse inp with delta'''
#     inp = inp.cpu().squeeze(0).numpy()
#     delta = delta.cpu().squeeze(0).numpy()
#     mask = mask.cpu().squeeze(0).numpy()
# 
#     n_chs, img_height, img_width = inp.shape
#     img_size = img_height * img_width
# 
#     inp = inp.reshape(n_chs, img_size)
#     delta = delta.reshape(n_chs, img_size)
#     # mask = mask.reshape(n_chs, img_size)
#     mask = mask.reshape(img_size)
# 
#     # mask = np.abs(mask).max(axis=0)
#     mask_idx = mask.argsort()[::-1]  # descend
#     protected_idx = mask_idx[:int(gamma * mask_idx.size)]
#     print('protected', protected_idx.size)
#     delta[:, protected_idx] = 0
# 
#     fused = np.clip(inp + delta, 0, 1)
#     fused = fused.reshape(n_chs, img_height, img_width)
#     fused = torch.FloatTensor(fused)
# 
#     protected = np.ones_like(inp)
#     protected[:, protected_idx] = 0
#     protected = protected.reshape(n_chs, img_height, img_width)
#     protected = torch.FloatTensor(protected)
#     return fused, protected


def run_hessian():
    sparse_args = {
        'lambda_t1': 1,
        'lambda_t2': 1,
        'lambda_l1': 0,
        'lambda_l2': 1e4,
        'n_iterations': 10,
        'optim': 'sgd',
        'lr': 1e-1,
    }

    configs = [
        ['resnet50', 'sparse', 'camshow', sparse_args],
        ['resnet50', 'vanilla_grad', 'camshow', None],
        ['resnet50', 'grad_x_input', 'camshow', None],
        ['resnet50', 'smooth_grad', 'camshow', None],
        ['resnet50', 'integrate_grad', 'camshow', None],
        # ['resnet50', 'deconv', 'imshow', None],
        # ['resnet50', 'guided_backprop', 'imshow', None],
        # ['resnet50', 'gradcam', 'camshow', None],
        # ['resnet50', 'excitation_backprop', 'camshow', None],
        # ['resnet50', 'contrastive_excitation_backprop', 'camshow', None],
    ]

    input_path = 'examples/tricycle.png'
    output_path = 'output/tricycle'
    model_name = 'resnet50'
    method_name = 'sparse'
    viz_style = 'camshow'
    raw_img = viz.pil_loader(input_path)
    transf = get_preprocess(model_name, method_name)
    model = utils.load_model(model_name)
    model.eval()
    model.cuda()

    # model_softplus = resnet50(pretrained=True)
    # model_softplus.eval()
    # model_softplus.cuda()

    attackers = [
        (NewHessianAttack(
            model,
            lambda_t1=0,
            lambda_t2=1,
            lambda_l1=0,
            lambda_l2=0,
            n_iterations=30,
            optim='sgd',
            lr=1e-2,
            epsilon=2 / 255
        ), 'gho'),
        (NoiseAttack(epsilon=2 / 255), 'rnd'),
    ]

    attacks = []
    for atk, attack_name in attackers:
        inp = utils.cuda_var(transf(raw_img).unsqueeze(0), requires_grad=True)
        delta = atk.attack(inp)
        attacks.append((delta, attack_name))

    rows = []
    for model_name, method_name, viz_style, kwargs in configs:
        inp_org = transf(raw_img)
        explainer = get_explainer(model, method_name, kwargs)

        filename_o = '{}.{}.{}.png'.format(output_path, method_name, 'org')
        # filename_m = '{}.{}.{}.png'.format(output_path, method_name, 'msk')
        saliency_org = get_saliency(model, explainer, inp_org, raw_img,
                                    model_name, method_name, viz_style,
                                    filename_o)

        row_viz = [method_name]
        row_viz.append('![]({})'.format(filename_o))
        # row_viz.append('![]({})'.format(filename_m))
        row_num = [method_name, ' ']
        for atk, attack_name in attacks:
            # inp_atk, protected = fuse(
            #     inp_org, atk.clone(), saliency_org,
            #     gamma=0.3)
            inp_atk = atk.clone()

            filename_a = '{}.{}.{}.png'.format(output_path,
                                               method_name,
                                               attack_name)
            saliency_atk = get_saliency(model, explainer, inp_atk.clone(),
                                        raw_img, model_name, method_name,
                                        viz_style, filename_a)

            # protected = utils.upsample(protected.unsqueeze(0),
            #                            (raw_img.height, raw_img.width))
            # protected = protected.cpu().numpy()
            # protected = np.abs(protected).max(axis=1).squeeze()
            # plt.imshow(protected, cmap='jet')
            # plt.axis('off')
            # plt.savefig(filename_m)

            corr = saliency_correlation(saliency_org, saliency_atk).correlation

            print(method_name, attack_name)
            print(get_prediction(model, inp_org.clone()).data.cpu().numpy()[0])
            print(get_prediction(model, inp_atk.clone()).data.cpu().numpy()[0])
            print(corr)
            print()

            row_viz.append('![]({})'.format(filename_a))
            row_num.append('Spearman: {}'.format(corr))

        rows.append(row_num)
        rows.append(row_viz)

    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = "ghorbani attack"
    writer.header_list = ['saliency method', 'original saliency']
    writer.header_list += [name for _, name in attacks]
    writer.value_matrix = rows
    with open('{}.ghorbani.md'.format(output_path), 'w') as f:
        writer.stream = f
        writer.write_table()


def run_hessian_full_validation():
    sparse_args = {
        'lambda_t1': 1,
        'lambda_t2': 1,
        'lambda_l1': 1e4,
        'lambda_l2': 1e4,
        'n_iterations': 10,
        'optim': 'sgd',
        'lr': 1e-1,
    }

    configs = [
        ['resnet50', 'sparse', 'camshow', sparse_args],
        # ['resnet50', 'vanilla_grad', 'camshow', None],
        # ['resnet50', 'grad_x_input', 'camshow', None],
        # ['resnet50', 'smooth_grad', 'camshow', None],
        # ['resnet50', 'integrate_grad', 'camshow', None],
        # ['resnet50', 'deconv', 'imshow', None],
        # ['resnet50', 'guided_backprop', 'imshow', None],
        # ['resnet50', 'gradcam', 'camshow', None],
        # ['resnet50', 'excitation_backprop', 'camshow', None],
        # ['resnet50', 'contrastive_excitation_backprop', 'camshow', None],
    ]

    model_name = 'resnet50'
    method_name = 'sparse'
    transf = get_preprocess(model_name, method_name)
    model = utils.load_model(model_name)
    model.eval()
    model.cuda()

    # model_softplus = resnet50(pretrained=True)
    # model_softplus.eval()
    # model_softplus.cuda()

    attackers = [
        (NewHessianAttack(
            model,
            lambda_t1=0,
            lambda_t2=1,
            lambda_l1=0,
            lambda_l2=0,
            n_iterations=30,
            optim='sgd',
            lr=1e-2,
            epsilon=2 / 255
        ), 'gho'),
        (NoiseAttack(epsilon=8 / 255), 'rnd'),
    ]

    image_path = '/fs/imageNet/imagenet/ILSVRC_val/'
    ghorbani_corr = [0] * len(configs)
    noise_corr = [0] * len(configs)
    total_count = 0.0
    num_images = 5
    for filename in glob.iglob(image_path + '**/*.JPEG', recursive=True):
        total_count += 1
        if total_count > num_images:
            continue
        raw_img = viz.pil_loader(filename)

        attacks = []
        for atk, attack_name in attackers:
            inp = utils.cuda_var(transf(raw_img).unsqueeze(0),
                                 requires_grad=True)
            delta = atk.attack(inp)
            attacks.append((delta, attack_name))

        for idx, (model_name, method_name, viz_style, kwargs) in enumerate(configs):
            inp_org = transf(raw_img)
            if method_name == 'sparse':
                explainer = get_explainer(model_softplus, method_name, kwargs)
            else:
                explainer = get_explainer(model, method_name, kwargs)

            saliency_org = get_saliency_no_viz(model, explainer, inp_org)

            for atk, attack_name in attacks:
                inp_atk, protected = fuse(
                    inp_org, atk.clone(), saliency_org,
                    gamma=0)
                saliency_atk = get_saliency_no_viz(model, explainer, inp_atk.clone())
                if attack_name == "gho":
                    ghorbani_corr[idx] += saliency_correlation(saliency_org, saliency_atk).correlation
                elif attack_name == "rnd":
                    noise_corr[idx] += saliency_correlation(saliency_org, saliency_atk).correlation

    for idx, c in enumerate(configs):
        print(c)
        print("Ghorbani Correlation: ", ghorbani_corr[idx] / num_images)
        print("Noise Correlation: ", noise_corr[idx] / num_images)

if __name__ == '__main__':
    run_hessian()
    # run_hessian_full_validation()
