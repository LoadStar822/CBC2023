# coding:utf-8
"""
Author  : Tian
Time    : 2023-06-02 19:31
Desc:
"""
import os

# 建立文件路径
ent_dir = r'D:\BaiduNetdiskDownload\SCOPe\filter_hydrogen'

# 遍历ent目录
for ent_file_name in os.listdir(ent_dir):
    ent_file = os.path.join(ent_dir, ent_file_name)
    if os.path.isfile(ent_file):
        with open(ent_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('REMARK  99 ASTRAL Source-PDB:'):
                seqid = line.split(': ')[1].strip()
                new_ent_file = seqid + '.ent'
                counter = 1
                while os.path.exists(os.path.join(ent_dir, new_ent_file)):
                    new_ent_file = seqid + '_' + str(counter) + '.ent'
                    counter += 1
                os.rename(ent_file, os.path.join(ent_dir, new_ent_file))
                break