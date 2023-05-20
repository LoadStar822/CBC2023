import os
import shutil

# 原始数据的路径
src_dir = 'D:\\BaiduNetdiskDownload\\SCOPe\\json'

# 新的目录结构的根目录
dst_root = 'D:\\BaiduNetdiskDownload\\SCOPe\\data\\scope-2.08-'

# 遍历原始目录中的所有文件
for filename in os.listdir(src_dir):
    # 假设文件名就是 seqid
    seqid = filename.split('.')[0]  # 移除 .json 后缀
    # 新的文件路径，按照 'scope-2.07-{seqid}/{seqid}/{filename}' 格式
    dst_dir = os.path.join(dst_root + seqid, seqid)
    dst_path = os.path.join(dst_dir, filename)
    # 创建新的目录
    os.makedirs(dst_dir, exist_ok=True)
    # 将文件从原始路径移动到新的路径
    shutil.move(os.path.join(src_dir, filename), dst_path)

print('Files reorganized successfully.')
