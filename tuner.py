import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from explainers_redo import SparseExplainer
from attack_hessian_redo import setup_imagenet, transf


class BatchTuner:

    def __init__(self):
        self.explainer = SparseExplainer()

    def tune(self, model, x):
        l1s = [0.5, 1, 10, 50, 100, 200, 500, 1000, 1500, 2000]
        l2s = [100, 200, 500, 1000, 2000, 5000, 1e4, 2e4, 5e4]
        batch_size, n_chs, height, width = x.shape
        assert batch_size == 1
        x_l1 = x.repeat(len(l1s), 1, 1, 1)
        self.explainer.lambda_l1 = Variable(torch.FloatTensor(l1s).cuda())
        saliency = self.explainer.explain(model, x_l1)
        return saliency


if __name__ == '__main__':
    model, batches = setup_imagenet(batch_size=1, n_examples=4)
    tuner = BatchTuner()
    batch = next(batches)
    ids, xs, ys = batch
    xs = torch.stack([transf(x) for x in xs]).cuda()
    print(xs.shape)
    saliency = tuner.tune(model, xs.clone())
    print(saliency.shape)
