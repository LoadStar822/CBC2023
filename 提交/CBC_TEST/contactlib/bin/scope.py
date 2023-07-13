# coding:utf-8
"""
Author  : Tian, Zhang
Time    : 2023-07-06 15:24
Desc:
"""
import json
import os
from collections import defaultdict

from Bio.PDB import PDBParser, PPBuilder


def process_file(path):
    basename = os.path.basename(path)
    pdbid, ext = os.path.splitext(basename)
    fn_ent = path

    dssp_path = path.replace('.ent', '.dssp')
    # os.system(f'contactlib/bin/mkdssp {fn_ent} {dssp_path}')
    os.system(f'/code/dssp/build/mkdssp {fn_ent} {dssp_path}')

    if not os.path.isfile(dssp_path):
        return

    with open(dssp_path, 'r') as file:
        dssp = file.readlines()

    data = defaultdict(dict)

    for line in dssp[28:]:
        idx = int(line[0:5])
        data['model'][idx] = {
            'res': line[13],
            'ss': line[16],
            'acc': float(line[34:38]),
            'nho0p': float(line[39:45]),
            'nho0e': float(line[46:50]),
            'ohn0p': float(line[50:56]),
            'ohn0e': float(line[57:61]),
            'nho1p': float(line[61:67]),
            'nho1e': float(line[68:72]),
            'ohn1p': float(line[72:78]),
            'ohn1e': float(line[79:83]),
            'phi': float(line[103:109]),
            'psi': float(line[109:115]),
            'x': float(line[115:122]),
            'y': float(line[122:129]),
            'z': float(line[129:136]),
        }

    # Extract sequence from PDB
    parser = PDBParser()
    structure = parser.get_structure('protein', fn_ent)
    ppb = PPBuilder()
    for pp in ppb.build_peptides(structure):
        sequence = pp.get_sequence()
    data['sequence'] = str(sequence)

    json_path = dssp_path.replace('.dssp', '.json')
    with open(json_path, 'w') as file:
        json.dump(data, file)
