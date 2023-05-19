# coding:utf-8
"""
Author  : Tian
Time    : 2023-05-19 12:53
Desc:
"""
import os
import shutil
from Bio import SeqIO

# CD-HIT输出文件路径
output_fasta = "D:\\BaiduNetdiskDownload\\SCOPe\\output.fasta"

# ent文件存放路径
input_ent_dir = "D:\\BaiduNetdiskDownload\\SCOPe\\filter"

# 筛选后的ent文件存放路径
output_ent_dir = "D:\\BaiduNetdiskDownload\\SCOPe\\filter_similarity"

# 从CD-HIT输出的fasta文件中读取序列ID
record_ids = [record.id for record in SeqIO.parse(output_fasta, "fasta")]

# 对每个ID，找到对应的.ent文件，然后将其复制到输出目录
for record_id in record_ids:
    input_ent_file = os.path.join(input_ent_dir, record_id + ".ent")
    output_ent_file = os.path.join(output_ent_dir, record_id + ".ent")

    # 检查输入文件是否存在
    if not os.path.isfile(input_ent_file):
        print(f"Warning: File {input_ent_file} does not exist.")
        continue

    # 移动文件
    shutil.move(input_ent_file, output_ent_file)