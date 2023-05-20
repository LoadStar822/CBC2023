# coding:utf-8
"""
Author  : Tian
Time    : 2023-05-20 17:18
Desc:
"""
import json
import os


def parse_remarks(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    remarks = [line for line in lines if line.startswith('REMARK')]

    for remark in remarks:
        if 'SCOPe-sccs' in remark:
            sccs = remark.split(': ')[1].strip()
            class_, fold, super_, family = sccs.split('.')
        elif 'Data-updated-release' in remark:
            release = remark.split(': ')[1].strip()

    remark_dict = {
        'class': class_,
        'fold': f'{class_}.{fold}',
        'super': f'{class_}.{fold}.{super_}',
        'family': f'{class_}.{fold}.{super_}.{family}',
        'release': '2.08'
    }
    return remark_dict


def add_remark_to_json(json_file, remark_dict):
    with open(json_file, 'r') as f:
        data = json.load(f)
    data['remark'] = remark_dict
    with open(json_file, 'w') as f:
        json.dump(data, f)


def process_dir(ent_dir, json_dir):
    for ent_file in os.listdir(ent_dir):
        if not ent_file.endswith('.ent'):
            continue
        ent_path = os.path.join(ent_dir, ent_file)
        remark_dict = parse_remarks(ent_path)

        json_file = ent_file.replace('.ent', '.json')
        json_path = os.path.join(json_dir, json_file)
        if os.path.exists(json_path):
            add_remark_to_json(json_path, remark_dict)


process_dir(r'D:\BaiduNetdiskDownload\SCOPe\filter_hydrogen', r'D:\BaiduNetdiskDownload\SCOPe\json')
