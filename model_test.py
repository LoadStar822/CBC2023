#!/opt/anaconda3/bin/python3 -BuW ignore
# coding: utf-8

import json
import h5py
import math
import random
import numpy as np
import torch as pt

from torch import nn
from torch.autograd import Variable
from glob import glob
from itertools import combinations
from scipy.spatial.distance import pdist, squareform

alphabet_res = 'LAGVESKIDTRPNFQYHMCW'  # 定义了一个由一组字母组成的字符串，这些字母是蛋白质残基的一种表示方式
alphabet_ss = 'HE TSGBI'  # 定义了一个由一组字母组成的字符串，这些字母是蛋白质二级结构的一种表示方式
radius = 8  # 定义了一个半径变量，值为8
diameter = radius * 2 + 1  # 定义了一个直径变量，其值是半径的两倍加1
volume = diameter * (diameter * 2 - 1)  # 定义了一个体积变量，其值是直径乘以(直径的两倍减1)
size0, size1, size2, size3 = 12, 1481, 2357, 5364  # 定义了四个大小变量，分别为12，1481，2357，5364


# 定义了一个名为listBatch的函数，该函数的作用是将数据桶(bucket)中的数据分成多个批次，每个批次包含不超过batchsize个元素
def listBatch(bucket, keyword, batchsize, batchlen=512):
    result = []  # 初始化结果列表
    bucket = sorted(bucket, key=lambda k: len(k[keyword]))  # 根据关键字keyword对bucket中的元素进行排序，排序的依据是每个元素的keyword的长度
    while (len(bucket) > 0):  # 当bucket中还有元素时执行循环
        batchsize = min([batchsize, len(bucket)])  # 确定批次大小，批次大小为batchsize和bucket中元素数量的最小值
        while len(bucket[batchsize - 1][keyword]) > batchlen:  # 当bucket中的元素的keyword的长度大于batchlen时执行循环
            batchsize, batchlen = (
                                              batchsize + 1) // 2, batchlen * 2  # 重新计算批次大小和批次长度，批次大小为(batchsize + 1) // 2，批次长度为batchlen * 2
        result.append(bucket[:batchsize])  # 将bucket中的前batchsize个元素作为一个批次，添加到结果列表中
        bucket = bucket[batchsize:]  # 更新bucket，将已经分配到批次的元素移除
    random.shuffle(result)  # 随机打乱结果列表中的元素
    return result  # 返回结果列表


# 定义了一个名为iterTrainFull的函数，该函数的作用是从给定的数据集中抽取批次的训练数据
def iterTrainFull(data, batchsize, bucketsize, noiserate=0.1):
    while True:  # 执行无限循环
        for batch in listBatch(random.sample(data, batchsize * bucketsize), 'coord',
                               batchsize):  # 从给定的数据集中随机抽取batchsize*bucketsize个样本，并将抽取的样本按照'coord'关键字分组为多个批次，每个批次的大小为batchsize
            sizemax = len(batch[-1]['coord'])  # 找出当前批次中'coord'长度最大的样本的长度
            noise = np.random.normal(0, noiserate, [len(batch), sizemax,
                                                    3])  # 生成一个形状为[len(batch), sizemax, 3]的正态分布噪声矩阵，均值为0，标准差为noiserate

            seq = np.zeros([len(batch), 1, sizemax, sizemax],
                           dtype=np.float32)  # 初始化一个形状为[len(batch), 1, sizemax, sizemax]的零矩阵，用于存放序列数据
            mask = np.zeros([len(batch)], dtype=np.int32)  # 初始化一个形状为[len(batch)]的零向量，用于存放mask数据
            label = np.zeros([len(batch), len(batch[-1]['label'])],
                             dtype=np.int64)  # 初始化一个形状为[len(batch), len(batch[-1]['label'])]的零矩阵，用于存放标签数据
            for i, b in enumerate(batch):  # 遍历当前批次中的每一个样本
                size = len(b['coord'])  # 获取当前样本的'coord'的长度

                seq[i, :, :size, :size] = squareform(
                    pdist(b['coord'] + noise[i, :size]))  # 计算当前样本的'00'加上对应的噪声后的两两之间的欧氏距离，并将距离矩阵保存到seq中对应的位置
                mask[i] = size  # 将当前样本的'coord'的长度保存到mask中对应的位置
                label[i] = b['label']  # 将当前样本的'label'保存到label中对应的位置
            seq[seq < 1.0] = 0.0  # 将seq中所有小于1.0的元素替换为0.0
            seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))  # 将seq中的nan元素替换为无穷大，并将seq转换为PyTorch张量
            mask = pt.from_numpy(mask)  # 将mask转换为PyTorch张量
            label = pt.from_numpy(label)  # 将label转换为PyTorch张量

            yield seq, mask, label  # 返回seq, mask, label三个变量



# 省略中间一些函数
def iterTestFull(data, batchsize):
    while True:  # 在无限循环中
        for batch in listBatch(data, 'coord', batchsize):  # 对于数据集中按'coord'分组并以batchsize大小划分的每一个批次
            sizemax = len(batch[-1]['coord'])  # 获取当前批次中'coord'最长的样本的长度

            # 初始化为0的张量，seq用于存放序列数据，mask用于存放mask数据，label用于存放标签数据
            seq = np.zeros([len(batch), 1, sizemax, sizemax], dtype=np.float32)
            mask = np.zeros([len(batch)], dtype=np.int32)
            label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
            for i, b in enumerate(batch):  # 遍历当前批次中的每个样本
                size = len(b['coord'])  # 获取当前样本的'coord'的长度

                # 计算样本中'coord'的两两之间的欧式距离，并保存到seq中
                seq[i, :, :size, :size] = squareform(pdist(b['coord']))
                mask[i] = size  # 将样本的'coord'的长度保存到mask中
                label[i] = b['label']  # 将样本的'label'保存到label中
            seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))  # 将seq中的nan元素替换为无穷大，并将seq转换为PyTorch张量
            mask = pt.from_numpy(mask)  # 将mask转换为PyTorch张量
            label = pt.from_numpy(label)  # 将label转换为PyTorch张量

            yield seq, mask, label  # 返回seq, mask, label三个变量


def iterTrainBond(data, batchsize, bucketsize, noiserate=0.1):
    while True:  # 在无限循环中
        # 对于从数据集中随机选择的batchsize*bucketsize个样本，按'hbond'分组并以batchsize大小划分的每一个批次
        for batch in listBatch(random.sample(data, batchsize * bucketsize), 'hbond', batchsize):
            sizemax = len(batch[-1]['hbond'])  # 获取当前批次中'hbond'最长的样本的长度
            noise = np.random.normal(0, noiserate, [len(batch), sizemax, diameter * 2, 3])  # 生成噪声

            # 初始化为0的张量，seq用于存放序列数据，mask用于存放mask数据，label用于存放标签数据
            seq = np.zeros([len(batch), sizemax, volume], dtype=np.float32)
            mask = np.ones([len(batch), sizemax], dtype=np.bool_)
            label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
            for i, b in enumerate(batch):  # 遍历当前批次中的每个样本
                size = len(b['hbond'])  # 获取当前样本的'hbond'的长度

                # 将样本中'hbond'的两两之间的欧式距离加上噪声，并保存到seq中
                seq[i, :size] = np.array([pdist(b['hbond'][j]['coord'] + noise[i, j]) for j in range(size)])
                mask[i, :size] = False  # 将mask的前size个元素设置为False
                label[i] = b['label']  # 将样本的'label'保存到label中
            seq[seq < 1.0] = 0.0  # 将seq中小于1.0的元素设置为0.0
            seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))  # 将seq中的nan元素替换为无穷大，并将seq转换为PyTorch张量
            mask = pt.from_numpy(mask)  # 将mask转换为PyTorch张量
            label = pt.from_numpy(label)  # 将label转换为PyTorch张量

            yield seq, mask, label  # 返回seq, mask, label三个变量


# 这个函数和上面的iterTrainBond函数非常类似，不同之处在于它使用了数据增强技术。数据增强是一种有效的提高模型泛化能力的方法，
# 它通过对原始数据进行各种随机变换（例如旋转、翻转、缩放等）来生成新的训练样本。
def iterAugmentBond(data, batchsize, bucketsize, noiserate=0.1):
    cache = h5py.File('cache/cache.hdf5', 'r')  # 打开名为'cache/cache.hdf5'的hdf5文件
    with open('astral/augment.json', 'r') as f:
        aug = json.load(f)  # 加载名为'astral/augment.json'的json文件
    for qid in aug.keys():  # 遍历aug中的每一个键
        todel = []
        for tid in aug[qid].keys():  # 遍历aug[qid]中的每一个键
            try:
                aug[qid][tid] = cache[tid]['hbond']  # 尝试从cache中获取tid对应的'hbond'数据，并赋值给aug[qid][tid]
            except KeyError:
                todel.append(tid)  # 如果发生KeyError异常，则将tid添加到todel列表中
        for tid in todel: del aug[qid][tid]  # 删除aug[qid]中键为todel中的每一个元素的项

    # 下面的代码和iterTrainBond函数中的代码几乎完全相同，只是在计算seq之前，会对当前样本的'hbond'数据进行增强
    while True:  # 在无限循环中
        for batch in listBatch(random.sample(data, batchsize * bucketsize), 'hbond', batchsize):
            for i, b in enumerate(batch):
                qid = b['pdbid']
                tid = random.choice(list(aug[qid].keys()))
                hbond = aug[qid][tid].value
                batch[i] = dict(hbond=hbond, label=b['label'])
            batch = sorted(batch, key=lambda k: len(k['hbond']))

            sizemax = len(batch[-1]['hbond'])
            noise = np.random.normal(0, noiserate, [len(batch), sizemax, diameter * 2, 3])

            seq = np.zeros([len(batch), sizemax, volume], dtype=np.float32)
            mask = np.ones([len(batch), sizemax], dtype=np.bool_)
            label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
            for i, b in enumerate(batch):
                size = len(b['hbond'])

                seq[i, :size] = np.array([pdist(b['hbond'][j] + noise[i, j]) for j in range(size)])
                mask[i, :size] = False
                label[i] = b['label']
            seq[seq < 1.0] = 0.0
            seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))
            mask = pt.from_numpy(mask)
            label = pt.from_numpy(label)

            yield seq, mask, label


# 这个函数和上面的iterTrainBond函数非常类似，只是它没有使用数据增强技术，而是直接对数据进行处理，然后返回处理后的数据。
def iterTestBond(data, batchsize):
    while True:  # 在无限循环中
        for batch in listBatch(data, 'hbond', batchsize):  # 对于数据集中按'hbond'分组并以batchsize大小划分的每一个批次
            sizemax = len(batch[-1]['hbond'])  # 获取当前批次中'hbond'最长的样本的长度

            # 初始化为0的张量，seq用于存放序列数据，mask用于存放mask数据，label用于存放标签数据
            seq = np.zeros([len(batch), sizemax, volume], dtype=np.float32)
            mask = np.ones([len(batch), sizemax], dtype=np.bool_)
            label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
            for i, b in enumerate(batch):  # 遍历当前批次中的每个样本
                size = len(b['hbond'])  # 获取当前样本的'hbond'的长度

                # 将样本中'hbond'的两两之间的欧式距离保存到seq中
                seq[i, :size] = np.array([pdist(b['hbond'][j]['coord']) for j in range(size)])
                mask[i, :size] = False  # 将mask的前size个元素设置为False
                label[i] = b['label']  # 将样本的'label'保存到label中
            seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))  # 将seq中的nan元素替换为无穷大，并将seq转换为PyTorch张量
            mask = pt.from_numpy(mask)  # 将mask转换为PyTorch张量
            label = pt.from_numpy(label)  # 将label转换为PyTorch张量

            yield seq, mask, label  # 返回seq, mask, label三个变量


# 这个函数用于评估模型的性能。它首先将输入数据和标签转换为Variable，并放到GPU上，然后将输入数据送入模型，得到模型的输出。
# 之后，它会根据模型的输出和标签计算准确率，然后返回准确率。
def eval(model, dataloader, datasize, ontology):
    result_acc, result_size = [], 0
    for x, m, y in dataloader:
        x = Variable(x).cuda()  # 将输入数据x转换为Variable，并放到GPU上
        m = Variable(m).cuda()  # 将mask数据m转换为Variable，并放到GPU上
        y0, y1, y2 = Variable(y[:, 0]).cuda(), Variable(y[:, 1]).cuda(), Variable(
            y[:, 2]).cuda()  # 将标签数据y转换为Variable，并放到GPU上
        _, _, yy2, _ = model(x, m)  # 将输入数据x和mask数据m送入模型，得到模型的输出
        yy2 = pt.argmax(yy2, dim=1)  # 对模型的输出进行argmax操作，得到预测的标签
        ontology['ontology01'] = ontology['ontology01'].cuda()
        ontology['ontology12'] = ontology['ontology12'].cuda()
        yy1 = pt.argmax(ontology['ontology12'][:, yy2].float(), dim=0).cuda()
        yy0 = pt.argmax(ontology['ontology01'][:, yy1].float(), dim=0).cuda()

        acc0 = (yy0 == y0)  # 计算预测标签yy0和真实标签y0的准确率
        acc01 = (yy1 == y1) & acc0  # 计算预测标签yy1和真实标签y1的准确率，并与acc0进行逻辑与运算
        acc012 = (yy2 == y2) & acc01  # 计算预测标签yy2和真实标签y2的准确率，并与acc01进行逻辑与运算

        # 将每一批次的准确率累加到result_acc中，将每一批次的样本数累加到result_size中
        result_acc.append([pt.sum(acc0).cpu().item(), pt.sum(acc01).cpu().item(), pt.sum(acc012).cpu().item()])
        result_size += len(y)

        # 如果已经处理的样本数大于等于指定的样本数，则跳出循环
        if result_size >= datasize:
            break

    return np.array(result_acc).sum(axis=0) / result_size  # 返回总的准确率


class DistBlock(nn.Module):  # 定义一个名为DistBlock的类，该类继承自nn.Module
    def __init__(self, dim):  # 定义类的初始化函数，接收一个参数dim
        super(DistBlock, self).__init__()  # 调用父类的初始化函数

        self.dim = dim  # 将dim保存为类的属性

    def forward(self, x):  # 定义类的前向计算函数，接收一个参数x
        x1 = x / 3.8;
        x2 = x1 * x1;
        x3 = x2 * x1  # 计算x的一次、二次和三次方，并将其分别保存到x1, x2, x3三个变量中
        xx = pt.cat([1 / (1 + x1), 1 / (1 + x2), 1 / (1 + x3)],
                    dim=self.dim).cuda()  # 计算1/(1+x1), 1/(1+x2), 1/(1+x3)三个张量，并在dim维度上将它们连接起来，将结果保存到xx变量中，并将xx变量转移到GPU上
        return xx  # 返回xx


class BaseNet(nn.Module):  # 定义一个名为BaseNet的类，该类继承自nn.Module
    def __init__(self, width, multitask=True):  # 定义类的初始化函数，接收两个参数width和multitask，默认值为True
        super(BaseNet, self).__init__()  # 调用父类的初始化函数

        self.multitask = multitask  # 将multitask保存为类的属性
        self.out0 = nn.Linear(width, size0)  # 定义一个全连接层，输入特征的数量为width，输出特征的数量为size0，将其保存为类的属性
        self.out1 = nn.Linear(width, size1)  # 定义一个全连接层，输入特征的数量为width，输出特征的数量为size1，将其保存为类的属性
        self.out2 = nn.Linear(width, size2)  # 定义一个全连接层，输入特征的数量为width，输出特征的数量为size2，将其保存为类的属性

    def forward(self, mem):  # 定义类的前向计算函数，接收一个参数mem
        if self.multitask:
            return self.out0(mem), self.out1(mem), self.out2(
                mem), mem  # 如果self.multitask为True，则分别使用self.out0, self.out1, self.out2全连接层处理mem，并返回处理后的结果以及原始的mem
        else:
            return None, None, self.out2(mem), mem  # 如果self.multitask为False，则只使用self.out2全连接层处理mem，并返回处理后的结果以及原始的mem


# 定义一个卷积块，继承自PyTorch的nn.Module类
class ConvBlock(nn.Module):
    # 初始化方法，参数cin表示输入通道数，cout表示输出通道数，dropout表示dropout的概率
    def __init__(self, cin, cout, dropout=0.1):
        # 调用父类nn.Module的初始化方法
        super(ConvBlock, self).__init__()

        # 定义一个2D卷积层，参数cin和cout分别表示输入通道数和输出通道数，卷积核大小5，步长2，padding值为2
        self.conv = nn.Conv2d(cin, cout, kernel_size=5, stride=2, padding=2)
        # 定义一个LayerNorm层，参数cout表示输出通道数
        self.norm = nn.LayerNorm(cout)
        # 定义一个序列化网络结构，包含一个2D的dropout层和ReLU激活层
        self.act = nn.Sequential(nn.Dropout2d(dropout), nn.ReLU())

    # 定义前向传播方法，参数x表示输入
    def forward(self, x):
        # 通过卷积层 -> 规范化层 -> 激活层的顺序进行前向传播，最后返回输出
        return self.act(self.norm(self.conv(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

# 定义DeepFold类，继承自nn.Module
class DeepFold(nn.Module):
    # 初始化方法，参数width表示宽度
    def __init__(self, width):
        # 调用父类nn.Module的初始化方法
        super(DeepFold, self).__init__()

        # 定义DistBlock网络模块，DistBlock是之前定义的一个网络模块，参数1表示输入通道数
        self.embed = DistBlock(1)

        # 定义卷积层的输出通道数序列
        cdim = [3, 64, 128, 256, 512, 512, width]
        # 创建多个卷积块，对应的输入通道数和输出通道数分别由cdim的相邻元素决定
        conv = [ConvBlock(cin, cout) for cin, cout in zip(cdim[:-1], cdim[1:])]
        # 将所有卷积块保存为一个ModuleList，使其能够被PyTorch正确处理
        self.conv = nn.ModuleList(conv)

        # 定义一个线性层，输入通道数为cdim的最后一个元素，输出通道数为size2，这里的size2需要在实际代码中定义
        self.out2 = nn.Linear(cdim[-1], size2)

    # 定义填充函数，参数data表示需要填充的数据，size表示需要填充的大小
    def masked_fill_(self, data, size):
        # 创建一个全1的Bool张量m，大小为[data.size(0), 1, data.size(-1)]
        m = pt.ones([data.size(0), 1, data.size(-1)]).bool().cuda()
        # 根据size的大小，将m中的部分值设为False
        for i, s in enumerate(size):
            m[i, :, :s] = False
        # 使用m作为掩码，将data中的相应位置设为0
        return data.masked_fill_(m.unsqueeze(2), 0).masked_fill_(m.unsqueeze(3), 0)

    # 定义前向传播方法，参数x表示输入，size表示输入的大小
    def forward(self, x, size):
        # 通过DistBlock网络模块进行前向传播，并进行填充处理
        mem = self.masked_fill_(self.embed(x), size)
        # 循环每一个卷积块，进行前向传播和填充处理
        for layer in self.conv:
            # 尺寸每次除以2
            size = (size + 1) // 2
            mem = self.masked_fill_(layer(mem), size)
        # 创建一个对角线上全为False，其他地方全为True的矩阵mask，大小与mem最后两维相同
        mask = pt.logical_not(pt.eye(mem.size(-1), dtype=pt.bool)).reshape([1, 1, *mem.shape[-2:]]).cuda()
        # 使用mask作为掩码，将mem中的相应位置设为0
        mem = mem.masked_fill_(mask, 0)
        # 计算mem在最后两维上的和，然后除以size，获得平均值
        mem = mem.sum(-1).sum(-1) / size.unsqueeze(1)
        # 返回None, None, 经过最后一个线性层的输出和mem
        return None, None, self.out2(mem), mem

# 定义一个神经块类，继承自PyTorch的nn.Module类
class NeuralBlock(nn.Module):
    # 初始化方法，参数nio表示输入和输出的神经元数量，dropout表示dropout的概率
    def __init__(self, nio, dropout=0.1):
        # 调用父类nn.Module的初始化方法
        super(NeuralBlock, self).__init__()

        # 定义一个序列化网络结构，包含dropout层，线性层，LayerNorm层，和ReLU激活层
        self.dense = nn.Sequential(nn.Dropout(dropout),
                                   nn.Linear(nio, nio), nn.LayerNorm(nio), nn.ReLU())

    # 定义前向传播方法，参数x表示输入
    def forward(self, x):
        # 将输入x通过dense序列网络，然后返回结果
        return self.dense(x)

# 定义一个神经网络类，继承自BaseNet，BaseNet需要在实际代码中定义
class NeuralNet(BaseNet):
    # 初始化方法，参数depth表示深度，width表示宽度，multitask表示是否进行多任务学习
    def __init__(self, depth, width, multitask=True):
        # 调用父类BaseNet的初始化方法
        super(NeuralNet, self).__init__(width, multitask=multitask)
        # 断言width能被64整除
        assert(width % 64 == 0)
        # nhead表示head的数量，ndense表示dense层的神经元数量
        nhead, ndense = width//64, width*4

        # 定义一个序列化网络结构，包含DistBlock网络模块，线性层，LayerNorm层，ReLU激活层，线性层，LayerNorm层，ReLU激活层
        self.embed = nn.Sequential(DistBlock(-1),
                                   nn.Linear(volume*3, ndense), nn.LayerNorm(ndense), nn.ReLU(),
                                   nn.Linear(ndense, width), nn.LayerNorm(width), nn.ReLU())

        # 定义一个序列化网络结构，包含depth个NeuralBlock
        self.encod = nn.Sequential(*[NeuralBlock(width) for i in range(depth)])

    # 定义前向传播方法，参数x表示输入，mask表示掩码
    def forward(self, x, mask):
        # 将输入x通过embed网络和encod网络，然后对结果进行mask处理，并计算每一行的平均值
        mem = self.encod(self.embed(x)).masked_fill_(mask.unsqueeze(2), 0)
        mem = mem.sum(1) / (mem.size(1) - mask.float().unsqueeze(2).sum(1))
        # 调用父类的forward方法，参数为mem
        return super().forward(mem)

# 下面的TransNet类和ConvTransNet类结构与上面类似，涉及到了Transformer的知识，主要差异在于它们使用了nn.TransformerEncoderLayer和nn.TransformerEncoder来构建神经网络模型。
# TransNet类是一个基于Transformer的网络类，ConvTransNet类是一个包含卷积层和Transformer的网络类，其他代码类似，这里就不一一进行详细解读和注释了。

# 定义一个TransNet类，它继承自BaseNet，BaseNet在实际代码中需要定义
class TransNet(BaseNet):
    # 初始化方法，参数depth表示深度，width表示宽度，multitask表示是否进行多任务学习
    def __init__(self, depth, width, multitask=True):
        # 调用父类BaseNet的初始化方法
        super(TransNet, self).__init__(width, multitask=multitask)
        # 断言width能被64整除
        assert(width % 64 == 0)
        # nhead表示head的数量，ndense表示dense层的神经元数量
        nhead, ndense = width//64, width*4

        # 定义一个序列化网络结构，包含DistBlock网络模块，线性层，LayerNorm层，ReLU激活层，线性层，LayerNorm层，ReLU激活层
        self.embed = nn.Sequential(DistBlock(-1),
                                   nn.Linear(volume*3, ndense), nn.LayerNorm(ndense), nn.ReLU(),
                                   nn.Linear(ndense, width), nn.LayerNorm(width), nn.ReLU())

        # 创建一个Transformer编码器层并添加到网络中
        layer_encod = nn.TransformerEncoderLayer(width, nhead, dim_feedforward=ndense, dropout=0.1)
        self.encod = nn.TransformerEncoder(layer_encod, depth)

    # 定义前向传播方法，参数x表示输入，mask表示掩码
    def forward(self, x, mask):
        # 将输入x通过embed网络，然后通过编码器，接着进行掩码处理，最后计算每一行的平均值
        mem = self.encod(self.embed(x).permute(1, 0, 2), src_key_padding_mask=mask).permute(1, 0, 2).masked_fill_(mask.unsqueeze(2), 0)
        mem = mem.sum(1) / (mem.size(1) - mask.float().unsqueeze(2).sum(1))
        # 调用父类的forward方法，参数为mem
        return super().forward(mem)


# 定义一个ConvTransNet类，它继承自BaseNet，BaseNet在实际代码中需要定义
class ConvTransNet(BaseNet):
    # 初始化方法，参数dconv表示卷积层的深度，dtrans表示Transformer编码器的深度，width表示宽度，multitask表示是否进行多任务学习
    def __init__(self, dconv, dtrans, width, multitask=True):
        # 调用父类BaseNet的初始化方法
        super(ConvTransNet, self).__init__(width, multitask=multitask)
        # 断言width能被64整除
        assert(width % 64 == 0)
        # nhead表示head的数量，ndense表示dense层的神经元数量
        nhead, ndense = width//64, width*4

        # 初始化DistBlock网络模块
        self.embed = DistBlock(1)

        # 创建多个卷积层并添加到网络中
        conv, cin, cout = [], 3, width // 2 ** (dconv - 1)
        for i in range(dconv):
            conv.append(ConvBlock(cin, cout))
            cin, cout = cout, min(cout*2, width)
        self.conv = nn.ModuleList(conv)

        # 创建一个Transformer编码器层并添加到网络中
        layer_encod = nn.TransformerEncoderLayer(width, nhead, dim_feedforward=ndense, dropout=0.1)
        self.encod = nn.TransformerEncoder(layer_encod, dtrans)

    # 定义一个masked_fill_方法，参数data表示数据，size表示大小
    def masked_fill_(self, data, size):
        # 初始化掩码
        m = pt.ones([data.size(0), 1, data.size(-1)]).bool().cuda()
        for i, s in enumerate(size):
            m[i, :, :s] = False
        # 返回经过掩码处理后的数据
        return data.masked_fill_(m.unsqueeze(2), 0).masked_fill_(m.unsqueeze(3), 0)

    # 定义前向传播方法，参数x表示输入，size表示大小
    def forward(self, x, size):
        # 对输入x进行embed处理和掩码处理
        mem = self.masked_fill_(self.embed(x), size)

        # 将数据通过卷积层处理，并进行掩码处理
        for layer in self.conv:
            size = (size + 1) // 2
            mem = self.masked_fill_(layer(mem), size)
        mask = pt.logical_not(pt.eye(mem.size(-1), dtype=pt.bool)).reshape([1, 1, *mem.shape[-2:]]).cuda()
        mem = mem.masked_fill_(mask, 0).sum(-1)

        # 创建掩码并进行掩码处理，然后通过编码器处理并进行掩码处理，最后计算每一行的平均值
        mask = pt.arange(mem.size(-1), dtype=pt.int32).repeat(mem.size(0), 1).cuda() >= size.unsqueeze(1)
        mem = self.encod(mem.permute(2, 0, 1), src_key_padding_mask=mask).permute(1, 0, 2).masked_fill_(mask.unsqueeze(2), 0)
        mem = mem.sum(1) / size.unsqueeze(1)

        # 调用父类的forward方法，参数为mem
        return super().forward(mem)


class new_model_augmentation(BaseNet):  # 定义一个新的模型类，该类继承自BaseNet
    def __init__(self, depth, width, multitask=True):  # 初始化函数，接收depth（TransformerEncoder层数）、width（特征维度）和multitask（是否进行多任务学习）三个参数
        super(new_model, self).__init__(width * 2, multitask=multitask)  # 调用父类BaseNet的初始化函数，传入宽度的两倍和multitask参数
        assert(width % 64 == 0)  # 断言宽度是64的倍数，如果不是，会抛出异常
        nhead, ndense = width//64, width*4  # 计算nhead和ndense的值，nhead等于宽度除以64的商，ndense等于宽度的四倍

        # 定义一个嵌入层，该层包括一个DistBlock，两个线性层，每个线性层后面都跟着一个层归一化（LayerNorm）和ReLU激活函数
        self.embed = nn.Sequential(DistBlock(-1),
                                   nn.Linear(volume*3, ndense), nn.LayerNorm(ndense), nn.ReLU(),
                                   nn.Linear(ndense, width), nn.LayerNorm(width), nn.ReLU())

        # 定义一个TransformerEncoderLayer，并用这个层来构建一个TransformerEncoder
        layer_encod = nn.TransformerEncoderLayer(width, nhead, dim_feedforward=ndense, dropout=0.1)
        self.encod = nn.TransformerEncoder(layer_encod, depth)

    def forward(self, x, mask, emd):  # 前向传播函数，接收x（输入数据）、mask（掩码）和emd（嵌入）三个参数
        # 使用嵌入层对输入数据x进行处理，然后用TransformerEncoder对处理后的数据进行编码
        # 然后对编码后的结果应用mask，然后对mask的结果进行求和和平均
        mem = self.encod(self.embed(x).permute(1, 0, 2), src_key_padding_mask=mask).permute(1, 0, 2).masked_fill_(mask.unsqueeze(2), 0)
        mem = mem.sum(1) / (mem.size(1) - mask.float().unsqueeze(2).sum(1))
        mem = pt.cat((mem,emd),1)  # 将mem和emd进行拼接
        return super().forward(mem)  # 调用父类BaseNet的前向传播函数，传入拼接后的mem
