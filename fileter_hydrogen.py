# coding:utf-8
"""
Author  : Tian
Time    : 2023-05-19 13:05
Desc:
"""
import os
import shutil
import subprocess
import re

# ent文件存放路径
input_ent_dir = "D:\\BaiduNetdiskDownload\\SCOPe\\filter_similarity"

# 筛选后的ent文件存放路径
output_ent_dir = "D:\\BaiduNetdiskDownload\\SCOPe\\filter_hydrogen"

# HBPLUS的路径
hbplus_exe = "E:\\学术\\生物信息学\\软件\\hbplus\\hbplus.exe"

# 对于输入目录中的每一个文件
for ent_file in os.listdir(input_ent_dir):
    # 完整的ent文件路径
    ent_file_path = os.path.join(input_ent_dir, ent_file)

    # 调用HBPLUS计算氢键
    hbplus_output = subprocess.run([hbplus_exe, ent_file_path], capture_output=True, text=True)

    # 从HBPLUS的输出中获取氢键数
    # 这次我们使用正则表达式匹配 "XXX hydrogen bonds found." 来获取氢键数
    match = re.search(r"(\d+) hydrogen bonds found.", hbplus_output.stdout)
    if match:
        h_bonds = int(match.group(1))

        # 如果氢键数大于等于20，将文件移动到输出目录
        if h_bonds >= 20:
            shutil.move(ent_file_path, os.path.join(output_ent_dir, ent_file))
