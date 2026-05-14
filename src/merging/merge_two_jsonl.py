import json

file1 = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/Rank-K-32B_dl20_rerank_depth_40_N_64_T_0.70_maxlen_16384_random_17638413794719791.jsonl"
file2 = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/Rank-K-32B_dl20_rerank_depth_40_N_64_T_0.70_maxlen_16384_random_17638500844306709.jsonl"
output = file2  # 写入到第二个文件

# 读取两个文件的所有行
all_lines = []

with open(file1, 'r', encoding='utf-8') as f:
    all_lines.extend([line.strip() for line in f if line.strip()])

with open(file2, 'r', encoding='utf-8') as f:
    all_lines.extend([line.strip() for line in f if line.strip()])

# 写入到输出文件
with open(output, 'w', encoding='utf-8') as f:
    for line in all_lines:
        f.write(line + '\n')

print(f"合并完成！总共 {len(all_lines)} 行")
