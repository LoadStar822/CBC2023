# coding:utf-8
"""
Author  : Tian
Time    : 2023-05-17 22:20
Desc:
"""
import os
from Bio import SeqIO

fasta_dir = 'D:\\BaiduNetdiskDownload\\SCOPe\\filter_fasta'
output_file = 'D:\\BaiduNetdiskDownload\\SCOPe\\combined.fasta'

# 列出所有的.fasta文件
fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith('.fasta')]

# 合并所有的fasta文件
with open(output_file, 'w') as outfile:
    for fname in fasta_files:
        with open(os.path.join(fasta_dir, fname)) as infile:
            for line in infile:
                outfile.write(line)
