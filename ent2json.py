# coding:utf-8
"""
Author  : Tian
Time    : 2023-05-20 11:03
Desc:
"""
import os
import json
import glob
from os.path import basename, splitext
from subprocess import call

ent_files_path = "/home/zqtianqinzhong/project/cbc/dataset/*.ent"
dssp_path = "/home/zqtianqinzhong/project/cbc/dssp"
json_path = "/home/zqtianqinzhong/project/cbc/json/"

for filename in glob.glob(ent_files_path):
    pdbid, ext = splitext(basename(filename))

    fn_ent = filename
    data = {}

    new_filename = os.path.join(json_path, pdbid + '.dssp')
    call([dssp_path, "-i", fn_ent, "-o", new_filename])

    if not os.path.isfile(new_filename):
        continue

    with open(new_filename, "r") as file:
        dssp = file.readlines()

    for line in dssp[28:]:
        idx = int(line[:5])
        data.setdefault('model', {})[idx] = {
            'res': line[13:14],
            'ss': line[16:17],
            'acc': int(float(line[34:38])),
            'nho0p': int(float(line[39:45])),
            'nho0e': int(float(line[46:50])),
            'ohn0p': int(float(line[50:56])),
            'ohn0e': int(float(line[57:61])),
            'nho1p': int(float(line[61:67])),
            'nho1e': int(float(line[68:72])),
            'ohn1p': int(float(line[72:78])),
            'ohn1e': int(float(line[79:83])),
            'phi': int(float(line[103:109])),
            'psi': int(float(line[109:115])),
            'x': int(float(line[115:122])),
            'y': int(float(line[122:129])),
            'z': int(float(line[129:136]))
        }

    new_filename = new_filename.replace('.dssp', '.json')
    with open(new_filename, "w") as file:
        file.write(json.dumps(data))
