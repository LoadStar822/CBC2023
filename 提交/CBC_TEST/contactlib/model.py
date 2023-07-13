# coding:utf-8
"""
Author  : Tian, Zhang
Time    : 2023-07-06 15:24
Desc:
"""

import random

import numpy as np
import torch as pt
import torch.nn.functional as F
from scipy.spatial.distance import pdist
from torch import nn
from torch.autograd import Variable

alphabet_res = 'LAGVESKIDTRPNFQYHMCW'
alphabet_ss = 'HE TSGBI'
radius = 8
diameter = radius * 2 + 1
volume = diameter * (diameter * 2 - 1)
size0, size1, size2, size3 = 12, 1481, 2357, 5364


def listBatch(bucket, keyword, batchsize, batchlen=512):
    result = []
    bucket = sorted(bucket, key=lambda k: len(k[keyword]))
    while (len(bucket) > 0):
        batchsize = min([batchsize, len(bucket)])
        while len(bucket[batchsize - 1][keyword]) > batchlen:
            batchsize, batchlen = (batchsize + 1) // 2, batchlen * 2
        result.append(bucket[:batchsize])
        bucket = bucket[batchsize:]
    random.shuffle(result)
    return result


def iterPredictBond(data, batchsize):
    while True:
        for batch in listBatch(data, 'hbond', batchsize):
            sizemax = len(batch[-1]['hbond'])
            seq = np.zeros([len(batch), sizemax, volume], dtype=np.float32)
            mask = np.ones([len(batch), sizemax], dtype=np.bool_)
            for i, b in enumerate(batch):
                size = len(b['hbond'])
                seq[i, :size] = np.array([pdist(b['hbond'][j]['coord']) for j in range(size)])
                mask[i, :size] = False
            seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))
            mask = pt.from_numpy(mask)
            emd = pt.stack([i["emd"] for i in batch])
            yield seq, mask, emd


def eval(model, dataloader, datasize, ontology):
    result_acc, result_size = [], 0
    for x, m, y in dataloader:
        x = Variable(x).cuda()
        m = Variable(m).cuda()
        y0, y1, y2 = Variable(y[:, 0]).cuda(), Variable(y[:, 1]).cuda(), Variable(y[:, 2]).cuda()
        _, _, yy2, _ = model(x, m)
        yy2 = pt.argmax(yy2, dim=1)
        yy1 = pt.argmax(ontology['ontology12'][:, yy2].float(), dim=0).cuda()
        yy0 = pt.argmax(ontology['ontology01'][:, yy1].float(), dim=0).cuda()
        acc0 = (yy0 == y0)
        acc01 = (yy1 == y1) & acc0
        acc012 = (yy2 == y2) & acc01
        acc0, acc01, acc012 = float(pt.mean(acc0.float())), float(pt.mean(acc01.float())), float(
            pt.mean(acc012.float()))
        result_acc.append([acc0, acc01, acc012, x.size(0)])
        result_size += x.size(0)
        if result_size >= datasize: break
    result_acc = np.array(result_acc)
    result_acc = np.sum(result_acc[:, :-1] * result_acc[:, -1:], axis=0) / np.sum(result_acc[:, -1]) * 100.0
    return (result_acc[0], result_acc[1], result_acc[2])


class DistBlock(nn.Module):
    def __init__(self, dim):
        super(DistBlock, self).__init__()
        self.dim = dim

    def forward(self, x):
        x1 = x / 3.8
        x2 = x1 * x1
        x3 = x2 * x1
        xx = pt.cat([1 / (1 + x1), 1 / (1 + x2), 1 / (1 + x3)], dim=self.dim).cuda()
        return xx


class BaseNet(nn.Module):
    def __init__(self, width, multitask=True):
        super(BaseNet, self).__init__()
        self.multitask = multitask
        self.out0 = nn.Linear(width, size0)
        self.out1 = nn.Linear(width, size1)
        self.out2 = nn.Linear(width, size2)

    def forward(self, mem):
        if self.multitask:
            return self.out0(mem), self.out1(mem), self.out2(mem), mem
        else:
            return None, None, self.out2(mem), mem


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dropout=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=5, stride=2, padding=2)
        self.norm = nn.LayerNorm(cout)
        self.act = nn.Sequential(nn.Dropout2d(dropout), nn.ReLU())

    def forward(self, x):
        return self.act(self.norm(self.conv(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))


class new_model_augmentation(BaseNet):
    def __init__(self, depth, width, multitask=True):
        super(new_model_augmentation, self).__init__(width * 2, multitask=multitask)
        assert (width % 64 == 0)
        nhead, ndense = width // 64, width * 4
        self.embed = nn.Sequential(DistBlock(-1),
                                   nn.Linear(volume * 3, ndense), nn.LayerNorm(ndense), nn.ReLU(),
                                   nn.Linear(ndense, width), nn.LayerNorm(width), nn.ReLU())
        layer_encod = nn.TransformerEncoderLayer(width, nhead, dim_feedforward=ndense, dropout=0.1)
        self.encod = nn.TransformerEncoder(layer_encod, depth)
        self.emd_attention = nn.Sequential(nn.Linear(width, width),
                                           nn.Tanh(),
                                           nn.Linear(width, 1))

    def forward(self, x, mask, emd):
        mem = self.encod(self.embed(x).permute(1, 0, 2), src_key_padding_mask=mask).permute(1, 0, 2).masked_fill_(
            mask.unsqueeze(2), 0)
        mem = mem.sum(1) / (mem.size(1) - mask.float().unsqueeze(2).sum(1))
        emd = emd.float()
        emd_weights = F.softmax(self.emd_attention(emd), dim=0)
        emd = (emd * emd_weights).sum(0)
        batch_size = mem.shape[0]
        emd = emd.unsqueeze(0).expand(batch_size, -1)
        mem = pt.cat((mem, emd), 1)
        return super().forward(mem)
