# coding:utf-8
"""
Author  : Tian, Zhang
Time    : 2023-07-07 15:24
Desc:
"""
import argparse
import json
import os
import re
import warnings
from glob import glob

from transformers import T5Config, T5Tokenizer, T5EncoderModel
from transformers.utils import logging

from ESMC.bin.scope import process_file
from ESMC.data import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--inputFile", type=str)
parser.add_argument("--gpuNumber", type=int, default=0)
argParameters = parser.parse_args()

filePath = argParameters.inputFile
gpuDevice = argParameters.gpuNumber
with open("ESMC/lab2idx.json", "r") as jsonFile:
    lab2idx = json.load(jsonFile)
idx2lab = dict(zip(lab2idx["super"].values(), lab2idx["super"].keys()))

process_file(filePath)

logging.set_verbosity_error()
warnings.filterwarnings('ignore')

newPath = filePath.split(".")[0] + "." + "json"

fileList = glob(newPath)
predictions, strId = [], []
computeDevice = pt.device('cuda:0' if pt.cuda.is_available() else 'cpu')
modelDirectory = "ESMC/model/prot_t5_xl_uniref50"
configModel = T5Config.from_pretrained(modelDirectory)
tokenizer = T5Tokenizer.from_pretrained(modelDirectory)
modelEncode = T5EncoderModel.from_pretrained(modelDirectory,
                                             config=configModel,
                                             ignore_mismatched_sizes=True).to(computeDevice)


def getEmbedding(sequence, model, tokenizer):
    length = len(sequence)
    sequence = [sequence]
    sequence = [" ".join(list(re.sub(r"[UZOB]", "X", sequenceVal))) for sequenceVal in sequence]

    tokenIds = tokenizer.batch_encode_plus(sequence, add_special_tokens=True, padding="longest")

    inputIds = pt.tensor(tokenIds['input_ids']).to(computeDevice)
    attentionMask = pt.tensor(tokenIds['attention_mask']).to(computeDevice)

    with pt.no_grad():
        embeddingRep = model(input_ids=inputIds, attention_mask=attentionMask)
    emb = embeddingRep.last_hidden_state[0, :length].mean(dim=0)
    return emb


try:
    for fn in fileList:
        structId = os.path.basename(fn)

        with open(fn, 'r') as fileHandle:
            scop = json.load(fileHandle)
        model = scop['model']
        size = len(model)
        dist2, coordinates = calDist2(model)
        frag = np.array([isFrag(model, dist2, i) for i in range(size)], dtype=np.bool_)
        frag2 = np.zeros([size, size], dtype=np.bool_)

        for index0, res0 in model.items():
            index0 = int(index0) - 1

            if float(res0['nho0e']) <= hbondmax:
                index1 = index0 + int(res0['nho0p'])
                if isFrag2(dist2, frag, index0, index1):
                    frag2[index0, index1] = True

            if float(res0['nho1e']) <= hbondmax:
                index2 = index0 + int(res0['nho1p'])
                if isFrag2(dist2, frag, index0, index2):
                    frag2[index0, index2] = True
        hbond = [buildContact(model, i, j) for i, j in zip(*np.where(frag2))]
        sequence = scop['sequence']
        emd = getEmbedding(sequence, modelEncode, tokenizer)
        predictions.append(dict(coord=coordinates, hbond=hbond, emd=emd))
        strId.append(structId[:-5])
        os.remove(fn)
except Exception as e:
    print(f"An error occurred: {e}")
try:
    modelFileName = 'ESMC/model_sav/modelnew_model_augmentation-seqid3-run1.pth'
    pt.cuda.set_device(int(gpuDevice))
    modelAugment = new_model_augmentation(depth=3, width=1024, multitask=True).cuda()
    modelAugment.load_state_dict(pt.load(modelFileName))
    modelAugment.eval()

    predictionLoader = iterPredictBond(predictions, 1)
    dataSize, resultSize = len(predictions), 0
    for x, m, e in predictionLoader:
        x = Variable(x).cuda()
        m = Variable(m).cuda()
        e = Variable(e).cuda()
        _, _, yy2, _ = modelAugment(x, m, e)
        yy2Idx = pt.argmax(yy2, dim=1)
        yy2Idx = yy2Idx.cpu().numpy()[0].tolist()
        yy2Lab = idx2lab[yy2Idx]
        givenId = strId[resultSize]
        print(yy2Lab)
        resultSize += x.size(0)
        if resultSize >= dataSize: break
except Exception as e:
    print(f"An error occurred: {e}")
