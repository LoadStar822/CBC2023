# coding:utf-8
"""
Author  : Tian, Zhang
Time    : 2023-07-07 15:24
Desc:
"""
import argparse
import json
import os
import warnings
from glob import glob

# pip install fair-esm
import esm
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
model_path = "ESMC/model/esm2_t36_3B_UR50D.pt"
model_lm, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
batch_converter = alphabet.get_batch_converter()
model_lm.eval()  # disables dropout for deterministic results
# configModel = T5Config.from_pretrained(modelDirectory)
# tokenizer = T5Tokenizer.from_pretrained(modelDirectory)
# modelEncode = T5EncoderModel.from_pretrained(modelDirectory,
#                                              config=configModel,
#                                              ignore_mismatched_sizes=True).to(computeDevice)
emd_length = 6144


def getEmbedding(sequence):
    if emd_length == 7680:
        sequences = [("protein1", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        layers_of_interest = [12, 24, 36]

        with pt.no_grad():
            results = model_lm(batch_tokens, repr_layers=layers_of_interest, return_contacts=True)

        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            layer_representations = []
            for layer in layers_of_interest:
                token_representations = results["representations"][layer]
                layer_representations.append(token_representations[i, 1: tokens_len - 1])
            sequence_representations.append(pt.cat(layer_representations, dim=-1))

        return sequence_representations[0].mean(0)
    elif emd_length == 6144:
        return


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
        emd = getEmbedding(sequence)
        predictions.append(dict(coord=coordinates, hbond=hbond, emd=emd))
        strId.append(structId[:-5])
        os.remove(fn)
except Exception as e:
    print(f"An error occurred: {e}")
try:
    modelFileName = 'ESMC/model_sav/6144维度 96 95 95.pth'
    pt.cuda.set_device(int(gpuDevice))
    modelAugment = ESMC(depth=3, width=1024, emd_length=emd_length, multitask=True).cuda()
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
