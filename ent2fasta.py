import os
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO


def write_fasta_from_pdb(pdb_file, fasta_file):
    # 解析PDB文件
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)

    # 提取蛋白质序列
    ppb = PPBuilder()
    for pp in ppb.build_peptides(structure):
        sequence = pp.get_sequence()

    # 保存为FASTA文件
    record = SeqRecord(Seq(str(sequence)),
                       id=os.path.basename(pdb_file).replace('.ent', ''),
                       description="")
    with open(fasta_file, "w") as output_handle:
        SeqIO.write(record, output_handle, "fasta")


ent_dir = 'D:\\BaiduNetdiskDownload\\SCOPe\\filter'
fasta_dir = 'D:\\BaiduNetdiskDownload\\SCOPe\\filter_fasta'

# 列出所有的.ent文件
ent_files = [f for f in os.listdir(ent_dir) if f.endswith('.ent')]

for ent_file in ent_files:
    ent_path = os.path.join(ent_dir, ent_file)
    fasta_file = os.path.join(fasta_dir, ent_file.replace('.ent', '.fasta'))

    # 跳过已经生成的fasta文件
    if os.path.exists(fasta_file):
        continue

    # 转换ent为fasta
    write_fasta_from_pdb(ent_path, fasta_file)
