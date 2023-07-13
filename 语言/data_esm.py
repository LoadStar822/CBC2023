#!/opt/anaconda3/bin/python3 -BuW ignore
# coding: utf-8

import sys
import json
import pickle
import math
import random
import numpy as np

from glob import glob
from os.path import basename
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform
from transformers import T5EncoderModel, T5Tokenizer, T5Config
import torch
import re
from model import *
import esm
from tqdm import tqdm
#
# seqid = sys.argv[1]
# countmin = int(sys.argv[2])
# hbondmax = float(sys.argv[3])

seqid = "esm_320"
countmin = 5
hbondmax = -0.5

model_type = "esm2_t6_8M_UR50D"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



if model_type == "prot_t5_xl_uniref50":
    model_path = "model/prot_t5_xl_uniref50"
    model_config = T5Config.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model_lm = T5EncoderModel.from_pretrained(model_path, config=model_config).to(device)
    model_lm.full() if device=='cpu' else model_lm.half()
elif model_type == "esm2_t6_8M_UR50D":
    model_path = "./model/ESM2/esm2_t6_8M_UR50D.pt"
    model_lm, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    batch_converter = alphabet.get_batch_converter()
    model_lm.eval()  # disables dropout for deterministic results
elif model_type == "esm2_t36_3B_UR50D":
    model_path = "./model/ESM2/esm2_t36_3B_UR50D.pt"
    model_lm, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    batch_converter = alphabet.get_batch_converter()
    model_lm.eval()  # disables dropout for deterministic results


def get_embedding(seq, model_type):
    if model_type == "Prot_T5_XL_Uniref50":
        l = len(seq)
        seq = [seq]
        seq = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq]

        ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding_rpr = model(input_ids=input_ids, attention_mask=attention_mask)
        emb = embedding_rpr.last_hidden_state[0,:l].mean(dim=0)
        return emb
    elif model_type == "esm2_t6_8M_UR50D":
        sequences = [("protein1", seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = model_lm(batch_tokens, repr_layers=[6], return_contacts=True)

        token_representations = results["representations"][6]

        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

        return sequence_representations[0]
    elif model_type == "esm2_t36_3B_UR50D":
        sequences = [("protein1", seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = model_lm(batch_tokens, repr_layers=[36], return_contacts=True)

        token_representations = results["representations"][36]

        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

        return sequence_representations[0]

def calDist2(model):
    size = max([int(i) for i in model.keys()])
    coord = np.ones((size, 3), dtype=np.float32) * np.inf
    for i in model.keys():
        ii = int(i) - 1
        coord[ii, 0] = model[i]['x']
        coord[ii, 1] = model[i]['y']
        coord[ii, 2] = model[i]['z']
    dist2 = np.nan_to_num(squareform(pdist(np.array(coord, dtype=np.float32))), nan=np.inf)
    return dist2, coord

def isFrag(model, dist2, idx, radius=radius, cutoff=4):
    if model.get(str(idx+1)) is None: return False
    if alphabet_res.find(model[str(idx+1)]['res']) < 0: return False
    for i in range(radius):
        ii = idx-i-1
        if ii >= 0:
            if model.get(str(ii+1)) is None: return False
            if alphabet_res.find(model[str(ii+1)]['res']) < 0: return False
            if dist2[ii, ii+1] > cutoff: return False

        jj = idx+i+1
        if jj < len(model):
            if model.get(str(jj+1)) is None: return False
            if alphabet_res.find(model[str(jj+1)]['res']) < 0: return False
            if dist2[jj, jj-1] > cutoff: return False
    return True

def isFrag2(dist2, frag, idx0, idx1, radius=radius):
    if abs(idx0 - idx1) <= 2: return False
    if not frag[idx0]: return False
    if not frag[idx1]: return False
    return True

def buildLabel(remark):
    label = [lab2idx['class'][remark['class']], lab2idx['fold'][remark['fold']],
             lab2idx['super'][remark['super']], lab2idx['family'][remark['family']]]
    label = np.array(label, dtype=np.int32)
    ontology01[label[0], label[1]] = ontology12[label[1], label[2]] = ontology23[label[2], label[3]] = True
    return label

def buildContact(model, idx0, idx1, radius=radius):
    res, ss, acc, dihedral, coord = [], [], [], [], []
    for i in list(range(idx0-radius, idx0+radius+1)) + list(range(idx1-radius, idx1+radius+1)):
        ii = str(i+1)
        if model.get(ii) is None:
            res.append(-1)
            ss.append(-1)
            acc.append(-1)
            dihedral.append([-360.0, -360.0])
            coord.append([np.inf, np.inf, np.inf])
        else:
            res.append(alphabet_res.find(model[ii]['res']))
            ss.append(alphabet_ss.find(model[ii]['ss']))
            acc.append(model[ii]['acc'])
            dihedral.append([model[ii]['phi'], model[ii]['psi']])
            coord.append([model[ii]['x'], model[ii]['y'], model[ii]['z']])
    res = np.array(res, dtype=np.int32)
    ss = np.array(ss, dtype=np.int32)
    acc = np.array(acc, dtype=np.float32)
    dihedral = np.array(dihedral, dtype=np.float32)
    coord = np.array(coord, dtype=np.float32)
    return dict(res=res, ss=ss, acc=acc, dihedal=dihedral, coord=coord)


print('#loading SCOP%s data ...' % seqid)
scop = {}
for fn in (glob('../data/json/*.json')):
    fid = basename(fn)[:7]
    with open(fn, 'r') as f:
        scop[fid] = json.load(f)
with open('../data/new_lab2idx.json', 'r') as f:
    lab2idx = json.load(f)
ontology01 = np.zeros([size0, size1], dtype=bool)
ontology12 = np.zeros([size1, size2], dtype=bool)
ontology23 = np.zeros([size2, size3], dtype=bool)
print('#size:', len(scop))

print('#building contactlib data ...')
data = []

for pdbid in tqdm(sorted(scop.keys(), key=lambda k: len(scop[k]['model']))):
    model = scop[pdbid]['model']
    size = len(model)
    if size < 20: continue

    dist2, coord = calDist2(model)
    frag = np.array([isFrag(model, dist2, i) for i in range(size)], dtype=bool)
    frag2 = np.zeros([size, size], dtype=bool)
    for idx0, res0 in model.items():
        idx0 = int(idx0)-1

        if float(res0['nho0e']) <= hbondmax:
            idx1 = idx0 + int(res0['nho0p'])
            if isFrag2(dist2, frag, idx0, idx1):
                frag2[idx0, idx1] = True

        if float(res0['nho1e']) <= hbondmax:
            idx2 = idx0 + int(res0['nho1p'])
            if isFrag2(dist2, frag, idx0, idx2):
                frag2[idx0, idx2] = True

    if np.sum(frag2) < 20: continue

    hbond = [buildContact(model, i, j) for i, j in zip(*np.where(frag2))]
    label = buildLabel(scop[pdbid]['remark'])
    release = float(scop[pdbid]['remark']['release'])
    sequence = scop[pdbid]['sequence']
    emd = get_embedding(sequence, model_type)
    data.append(dict(coord=coord, hbond=hbond, label=label, pdbid=pdbid, release=release, emd = emd))
print('#size:', len(data))
print('#ontology:', np.sum(ontology01), np.sum(ontology12), np.sum(ontology23))

with open('../data/ontology%s.sav' % seqid, 'wb') as f:
    pickle.dump(dict(ontology01=ontology01, ontology12=ontology12, ontology23=ontology23), f)

print('#splitting train-valid-test data ...')
train, valid, test = [], [], []
random.shuffle(data)
count = {}
for d in data:
    l = d['label'][2]
    if count.get(l, 0) < countmin:
        train.append(d)
    else:
        valid.append(d)
    count[l] = count.get(l, 0) + 1
    # if d['release'] < 2.08:
    #     if count.get(l, 0) < countmin: train.append(d)
    #     else: valid.append(d)
    #     count[l] = count.get(l, 0) + 1
    # else:
    #     test.append(d)
trainext, valid = train_test_split(valid, test_size=0.4)
train.extend(trainext)
test, valid = train_test_split(valid, test_size=0.5)

random.shuffle(train)
test = [d for d in test if count.get(d['label'][2], 0) >= countmin]
print('#size:', len(train), len(valid), len(test))
train_small, val_small, test_small, other = [],[],[],[]
train_small, other = train_test_split(train, test_size=0.8)
val_small, other = train_test_split(valid, test_size=0.8)
test_small, other = train_test_split(test, test_size=0.8)

with open('../data/train%s.sav' % seqid, 'wb') as f:
    pickle.dump(train, f)
with open('../data/train_small%s.sav' % seqid, 'wb') as f:
    pickle.dump(train_small, f)
with open('../data/valid%s.sav' % seqid, 'wb') as f:
    pickle.dump(valid, f)
with open('../data/valid_small%s.sav' % seqid, 'wb') as f:
    pickle.dump(val_small, f)
with open('../data/test%s.sav' % seqid, 'wb') as f:
    pickle.dump(test, f)
with open('../data/test_small%s.sav' % seqid, 'wb') as f:
    pickle.dump(test_small, f)

print('#done!!!')

