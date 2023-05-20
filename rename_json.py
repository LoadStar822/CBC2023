# coding:utf-8
"""
Author  : Tian
Time    : 2023-05-20 20:54
Desc:
"""
import os
import json

# 建立文件路径
json_dir = r'D:\BaiduNetdiskDownload\SCOPe\json'
ent_dir = r'D:\BaiduNetdiskDownload\SCOPe\filter_hydrogen'

# 遍历json目录
for json_file in os.listdir(json_dir):
    # 判断是否是文件
    if os.path.isfile(os.path.join(json_dir, json_file)):
        # 打开相应的ent文件
        ent_file = os.path.join(ent_dir, json_file.split('.')[0] + '.ent')
        if os.path.isfile(ent_file):
            with open(ent_file, 'r') as f:
                for line in f.readlines():
                    if line.startswith('REMARK  99 ASTRAL Source-PDB:'):
                        seqid = line.split(': ')[1].strip()  # 取出seqid
                        new_json_file = seqid + '.json'  # 创建新的json文件名
                        counter = 1
                        # 确保新文件名是唯一的
                        while os.path.exists(os.path.join(json_dir, new_json_file)):
                            new_json_file = seqid + '_' + str(counter) + '.json'  # 添加后缀
                            counter += 1
                        os.rename(os.path.join(json_dir, json_file), os.path.join(json_dir, new_json_file))  # 重命名json文件
                        break

