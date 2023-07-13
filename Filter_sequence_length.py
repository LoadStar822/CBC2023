import os
import shutil
from Bio.PDB import PDBParser, PPBuilder

# 创建一个PDB解析器
parser = PDBParser()

# 创建一个蛋白质多肽构建器，用于从结构中提取氨基酸序列
ppb = PPBuilder()

# 指定你的PDB文件目录和过滤后的目录
pdb_dir = "D:/BaiduNetdiskDownload/alphafold"
filtered_dir = "D:/BaiduNetdiskDownload/alphafold_filter"

# 如果过滤后的目录不存在，创建它
if not os.path.exists(filtered_dir):
    os.makedirs(filtered_dir)

# 递归遍历目录及其所有子目录中的文件
for root, dirs, files in os.walk(pdb_dir):
    for pdb_file in files:
        if pdb_file.endswith(".pdb"):
            try:
                # 使用解析器读取PDB文件
                structure = parser.get_structure(pdb_file[:-4], os.path.join(root, pdb_file))
            except Exception as e:
                print(f"Error while parsing {pdb_file}: {e}")
                continue

            # 从结构中提取氨基酸序列
            moved = False
            for model in structure:
                for chain in model:
                    for pp in ppb.build_peptides(chain):
                        seq = pp.get_sequence()

                        # 如果序列长度大于等于20，移动文件到过滤后的目录
                        if len(seq) >= 20:
                            shutil.move(os.path.join(root, pdb_file), os.path.join(filtered_dir, pdb_file))
                            moved = True
                            break
                    if moved:
                        break
                if moved:
                    break
