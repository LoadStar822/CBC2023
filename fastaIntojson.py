# coding:utf-8
"""
Author  : Tian
Time    : 2023-06-07 15:24
Desc:
"""
import os
import json
from Bio import SeqIO

fasta_dir = 'D:/BaiduNetdiskDownload/SCOPe/filter_hydrogen_fasta/'
json_dir = 'D:/BaiduNetdiskDownload/SCOPe/json/'

# 遍历fasta文件夹
for fasta_file in os.listdir(fasta_dir):
    if fasta_file.endswith('.fasta'):
        # 读取fasta文件中的序列
        fasta_path = os.path.join(fasta_dir, fasta_file)
        for record in SeqIO.parse(fasta_path, 'fasta'):
            sequence = str(record.seq)

        # 找到对应的json文件
        json_file = fasta_file.replace('.fasta', '.json')
        json_path = os.path.join(json_dir, json_file)

        # 打开json文件，如果没有'sequence'键，添加序列信息，并保存
        with open(json_path, 'r') as jsonf:
            data = json.load(jsonf)

        if 'sequence' not in data:
            data['sequence'] = sequence

            with open(json_path, 'w') as jsonf:
                json.dump(data, jsonf)
