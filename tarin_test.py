#!/opt/anaconda3/bin/python3 -BuW ignore
# 声明使用哪个Python解释器来运行此脚本。并且在脚本运行时，将缓冲设为非缓冲模式，并忽略警告

# coding: utf-8
# 声明脚本的字符编码为utf-8

import os
import sys
import time
import pickle
import numpy as np
import torch
import torch as pt
# 引入所需的库

from torch import nn, optim
from torch.autograd import Variable
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
# 从各个库中引入所需的具体模块

from model import *
# 从model.py文件中引入所有内容，包括定义的模型类等

modid = sys.argv[1]
seqid = sys.argv[2]
runid = sys.argv[3]
devid = int(sys.argv[4])
batchsize = 16
# 获取脚本运行时的输入参数，并设置batch size的大小

pt.cuda.set_device(devid)
print('#cuda devices:', devid)
# 设置使用哪个GPU设备，并打印设备编号

print('#loading data ...')
# 开始加载数据，包括训练集、验证集、测试集和本体数据，并将本体数据转化为PyTorch的tensor形式
with open('data/train%s.sav' % seqid, 'rb') as f:
    train = pickle.load(f)
with open('data/valid%s.sav' % seqid, 'rb') as f:
    valid = pickle.load(f)
with open('data/test%s.sav' % seqid, 'rb') as f:
    test = pickle.load(f)
with open('data/ontology%s.sav' % seqid, 'rb') as f:
    ontology = pickle.load(f)
    for k in ontology.keys():
        ontology[k] = pt.from_numpy(ontology[k])
modelfn = 'output/model%s-seqid%s-run%s.pth' % (modid, seqid, runid)
# 设置模型文件的路径
print('##size:', len(train), len(valid), len(test))
# 打印训练集、验证集和测试集的大小

print('#building model ...')
# 开始构建模型
if modid == 'ContactLib-ATT01':
    model = TransNet(depth=3, width=1024, multitask=True).cuda()
    model.load_state_dict(torch.load(modelfn))
    model.train()  # 确保模型处于训练模式
    trainloader = iterTrainBond(train, batchsize, 97)
    validloader = iterTestBond(valid, batchsize)
    testloader = iterTestBond(test, batchsize)
    # 如果模型的标识符为'ContactLib-ATT01'，则构建TransNet模型，并将其加载到指定的GPU设备上
    # 如果存在之前训练过的模型，就加载模型参数
    # 设置模型为训练模式
    # 创建训练集、验证集和测试集的迭代器

print('##size:', np.sum(np.fromiter((p.numel() for p in model.parameters() if p.requires_grad), dtype=int)))
# 打印模型的参数数量

print('#training model ...')
# 开始训练模型
lr_init, lr_min, epochiter, epochstop = 1e-3, 1e-5, 64, 2
# 设置初始学习率、最小学习率、每次迭代的轮数和停止迭代的轮数
opt = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9)
# 定义优化器，这里使用随机梯度下降优化器SGD，设置学习率和动量
sched0 = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2, eta_min=lr_min)
sched1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=epochiter, T_mult=1, eta_min=lr_min)
# 定义学习率调度器，这里使用余弦退火策略

best_acc, best_epoch = 0, 0
# 初始化最佳准确率和最佳准确率所在的轮次

# 开始训练过程
for epoch in range(epochiter * 16 - 1):
    t0 = time.perf_counter()
    # 获取当前时间

    model.train()
    train_loss, batch_size = [], 0
    # 将模型设置为训练模式，并初始化训练损失和批次大小
    for i, (x, m, y) in enumerate(trainloader):
        x = Variable(x).cuda()
        m = Variable(m).cuda()
        y0, y1, y2 = Variable(y[:, 0]).cuda(), Variable(y[:, 1]).cuda(), Variable(y[:, 2]).cuda()
        idx2 = combinations(range(x.size(0)), 2)
        yy0, yy1, yy2, yycode = model(x, m)
        # 从训练集的迭代器中获取数据，并将数据加载到GPU设备上，然后通过模型得到预测结果

        loss, losssize = nn.functional.cross_entropy(yy2, y2), 1
        if yy1 is not None: loss, losssize = loss + nn.functional.cross_entropy(yy1, y1), losssize + 1
        if yy0 is not None: loss, losssize = loss + nn.functional.cross_entropy(yy0, y0), losssize + 1
        loss = loss / losssize + pt.sqrt(pt.mean(pt.square(yycode))) * 0.1
        opt.zero_grad()
        loss.backward()
        train_loss.append([float(loss), x.size(0)])
        # 计算损失函数，这里使用的是交叉熵损失函数，并考虑了模型的多任务学习
        # 然后使用优化器将梯度清零，计算梯度，并将损失添加到训练损失中

        opt.step()
        # 使用优化器更新模型参数

        if epoch + 1 < epochiter:
            sched0.step(epoch + batch_size / len(train))
        else:
            sched1.step(epoch + 1 + batch_size / len(train))
        batch_size += x.size(0)
        if batch_size >= len(train): break
    train_loss = np.array(train_loss)
    train_loss = np.sum(train_loss[:, 0] * train_loss[:, 1]) / np.sum(train_loss[:, 1])
    # 根据调度器调整学习率，并更新批次大小
    # 计算训练损失的平均值

    model.eval()
    valid_acc = eval(model, validloader, len(valid), ontology)
    # 将模型设置为评估模式，并在验证集上评估模型的性能

    if valid_acc[-1] > best_acc:
        test_acc = eval(model, testloader, len(test), ontology)
        summary = [opt.param_groups[0]['lr'], train_loss, *valid_acc, *test_acc, time.perf_counter() - t0]
        print('#epoch[%d]:\t%.1e\t%.5f\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.1fs\t*' % (epoch + 1, *summary))
        directory = os.path.dirname(modelfn)

        if not os.path.exists(directory):
            os.makedirs(directory)
        pt.save(model.state_dict(), modelfn)
        best_acc, best_epoch = valid_acc[-1], epoch
        # 如果验证集的准确率超过了之前的最佳准确率，则在测试集上评估模型的性能，并打印训练的结果
        # 然后将模型的参数保存到指定的文件中，更新最佳准确率和最佳准确率所在的轮次
    else:
        summary = [opt.param_groups[0]['lr'], train_loss, *valid_acc, time.perf_counter() - t0]
        print('#epoch[%d]:\t%.1e\t%.5f\t%.2f%%\t%.2f%%\t%.2f%%\t%.1fs' % (epoch + 1, *summary))
        # 如果验证集的准确率没有超过之前的最佳准确率，则打印训练的结果
    if (epoch + 1) % epochiter == 0 and (epoch - best_epoch) // epochiter >= epochstop: break
    # 如果已经达到了预定的轮次，并且超过了预定的轮次没有改进，那么就停止训练

print('#done!!!')
# 打印训练完成的信息
