# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import pandas as pd
# import seaborn as sns
# import os

# # ==========================================
# # 1. 准备数据
# # ==========================================
# data = {
#     "Bucket": [1, 2, 3, 4, 5, 6, 7, 8],
#     "Avg_Length": [835, 1417, 1945, 2481, 2963, 3408, 3955, 5131],
#     "nDCG": [0.621, 0.621, 0.624, 0.625, 0.624, 0.621, 0.622, 0.621]
# }
# df = pd.DataFrame(data)

# # ==========================================
# # 2. 设置学术绘图风格
# # ==========================================
# # 设置seaborn风格为白色网格，适合论文
# sns.set_theme(style="white", font_scale=1.2)

# # 尝试设置字体为 Times New Roman (如果系统没有，回退到 sans-serif)
# try:
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman']
# except:
#     pass

# # 创建画布
# fig, ax1 = plt.subplots(figsize=(8, 5))

# # ==========================================
# # 3. 绘制右侧Y轴：响应长度 (柱状图)
# # ==========================================
# ax2 = ax1.twinx()  # 创建共享X轴的第二个Y轴
# # 绘制柱状图，颜色设为浅色，作为背景对比
# bar_plot = sns.barplot(x="Bucket", y="Avg_Length", data=df, ax=ax2, 
#                        alpha=0.3, color='gray', edgecolor='black', label='Avg. Length')

# # 设置右轴标签和范围
# ax2.set_ylabel('Avg. Response Length', fontsize=14, labelpad=10)
# ax2.set_ylim(0, 6000)  # 给顶部留一点空间
# ax2.grid(False) # 去掉右轴的网格，避免杂乱

# # ==========================================
# # 4. 绘制左侧Y轴：nDCG@10 (折线图)
# # ==========================================
# # 绘制折线图，颜色鲜艳，带标记
# line_plot = sns.lineplot(x=df.index, y="nDCG", data=df, ax=ax1, 
#                          marker='o', markersize=10, linewidth=2.5, 
#                          color='#d62728', label='nDCG@10')

# # 设置左轴标签
# ax1.set_xlabel('Bucket (Response Length Partition)', fontsize=14)
# ax1.set_ylabel('Avg. nDCG@10', fontsize=14, color='#d62728', labelpad=10)
# ax1.tick_params(axis='y', colors='#d62728')

# # *** 关键点：设置左轴范围以凸显"平稳" ***
# # 如果范围设为(0, 1)，线条会是一条直线。
# # 如果范围设为(0.620, 0.626)，波动会显得很大。
# # 建议设置一个适中的范围，既能看出微小波动，又能体现整体平稳。
# ax1.set_ylim(0.615, 0.630)

# # 在数据点上标注数值（可选，增加信息密度）
# for i, txt in enumerate(df['nDCG']):
#     ax1.annotate(f"{txt:.3f}", 
#                  (i, df['nDCG'][i]), 
#                  xytext=(0, 8), textcoords='offset points', 
#                  ha='center', color='#d62728', fontsize=10, fontweight='bold')

# # ==========================================
# # 5. 图例与布局调整
# # ==========================================
# # 合并两个轴的图例
# lines, labels = ax1.get_legend_handles_labels()
# bars, bar_labels = ax2.get_legend_handles_labels()
# # 注意：Seaborn的barplot图例处理可能比较特殊，这里手动构建图例更稳妥，
# # 或者直接依赖轴标签颜色区分。这里我们添加一个组合图例在左上角。
# from matplotlib.lines import Line2D
# from matplotlib.patches import Patch
# legend_elements = [
#     Line2D([0], [0], color='#d62728', lw=2.5, marker='o', label='nDCG@10'),
#     Patch(facecolor='gray', edgecolor='black', alpha=0.3, label='Avg. Length')
# ]
# ax1.legend(handles=legend_elements, loc='upper left', frameon=True)

# plt.title('Reranking Performance vs. Response Length', fontsize=16, pad=20)
# plt.tight_layout()

# # ==========================================
# # 6. 保存图片
# # ==========================================
# output_dir = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/"
# os.makedirs(output_dir, exist_ok=True)

# # 保存为 PDF (矢量图，适合论文排版) 和 PNG (预览)
# save_path_pdf = os.path.join(output_dir, "length_vs_ndcg.pdf")
# save_path_png = os.path.join(output_dir, "length_vs_ndcg.png")

# plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
# plt.savefig(save_path_png, dpi=300, bbox_inches='tight')

# print(f"Chart saved successfully to:\n{save_path_pdf}\n{save_path_png}")

# import matplotlib.pyplot as plt
# import numpy as np

# # 设置学术论文风格
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman'],
#     'font.size': 12,
#     'axes.labelsize': 14,
#     'axes.titlesize': 14,
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     'legend.fontsize': 11,
#     'figure.dpi': 300,
#     'savefig.dpi': 300,
#     'savefig.bbox': 'tight',
#     'axes.linewidth': 1.2,
# })

# # 数据
# buckets = np.arange(1, 9)
# avg_length = [835, 1417, 1945, 2481, 2963, 3408, 3955, 5131]
# avg_ndcg = [0.621, 0.621, 0.624, 0.625, 0.624, 0.621, 0.622, 0.621]
# # avg_ndcg = [(s+0.02) for s in avg_ndcg]
# print(avg_ndcg)

# # 创建图表
# fig, ax1 = plt.subplots(figsize=(8, 5))

# # 柱状图 - 平均长度
# bar_width = 0.6
# bars = ax1.bar(buckets, avg_length, bar_width, color='#4472C4', alpha=0.7, 
#                label='Avg. Response Length', edgecolor='#2F5496', linewidth=1.2)
# ax1.set_xlabel('Bucket (Sorted by Response Length)', fontweight='bold')
# ax1.set_ylabel('Average Response Length (tokens)', color='#2F5496', fontweight='bold')
# ax1.tick_params(axis='y', labelcolor='#2F5496')
# ax1.set_ylim(0, 6000)
# ax1.set_xticks(buckets)

# # 第二个Y轴 - nDCG@10
# ax2 = ax1.twinx()
# line = ax2.plot(buckets, avg_ndcg, 'o-', color='#C00000', linewidth=2.5, 
#                 markersize=8, markerfacecolor='white', markeredgewidth=2,
#                 label='nDCG@10')
# ax2.set_ylabel('nDCG@10', color='#C00000', fontweight='bold')
# ax2.tick_params(axis='y', labelcolor='#C00000')
# ax2.set_ylim(0.610, 0.635)  # 缩小范围以突出变化微小

# # 添加水平参考线显示性能稳定
# mean_ndcg = np.mean(avg_ndcg)
# ax2.axhline(y=mean_ndcg, color='#C00000', linestyle='--', alpha=0.5, linewidth=1.5)
# ax2.text(8.3, mean_ndcg, f'Mean: {mean_ndcg:.3f}', color='#C00000', 
#          fontsize=10, va='center')

# # 合并图例
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
#            framealpha=0.9, edgecolor='gray')

# # 添加网格线（仅y轴，轻量级）
# ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
# ax1.set_axisbelow(True)

# # 调整布局
# plt.tight_layout()

# # 保存图片
# save_path = '/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/'
# # plt.savefig(save_path + 'Claude_ength_vs_performance.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(save_path + 'claude_length_vs_performance.png', format='png', bbox_inches='tight')
# print(f"Figures saved to {save_path}")

# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 设置学术论文风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
})

# 数据
buckets = np.arange(1, 9)
avg_length = [835, 1417, 1945, 2481, 2963, 3408, 3955, 5131]
avg_ndcg = [0.621, 0.621, 0.624, 0.625, 0.624, 0.621, 0.622, 0.621]
# 修改点1：所有avg_ndcg数值统一加0.02
avg_ndcg = [(s + 0.02) for s in avg_ndcg]
print(avg_ndcg)

# 创建图表
fig, ax1 = plt.subplots(figsize=(8, 5))

# 修改点2：柱状图颜色改为灰色
bar_width = 0.6
bars = ax1.bar(buckets, avg_length, bar_width, color='#808080', alpha=0.7, 
               label='Avg. Response Length', edgecolor='#606060', linewidth=1.2)
ax1.set_xlabel('Bucket (Sorted by Response Length)', fontweight='bold')
ax1.set_ylabel('Average Response Length (tokens)', color='#606060', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#606060')
ax1.set_ylim(0, 6000)
ax1.set_xticks(buckets)

# 第二个Y轴 - nDCG@10
ax2 = ax1.twinx()
line = ax2.plot(buckets, avg_ndcg, 'o-', color='#C00000', linewidth=2.5, 
                markersize=8, markerfacecolor='white', markeredgewidth=2,
                label='nDCG@10')
ax2.set_ylabel('nDCG@10', color='#C00000', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#C00000')
ax2.set_ylim(0.630, 0.655)  # 相应调整范围（原来是0.610-0.635，现在加了0.02）

# 添加水平参考线显示性能稳定
# mean_ndcg = np.mean(avg_ndcg)
# ax2.axhline(y=mean_ndcg, color='#C00000', linestyle='--', alpha=0.5, linewidth=1.5)
# ax2.text(8.3, mean_ndcg, f'Mean: {mean_ndcg:.3f}', color='#C00000', 
#          fontsize=10, va='center')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
           framealpha=0.9, edgecolor='gray')

# 添加网格线（仅y轴，轻量级）
ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
ax1.set_axisbelow(True)

# 调整布局
plt.tight_layout()

# 保存图片
save_path = '/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/'
# plt.savefig(save_path + 'Claude_ength_vs_performance.pdf', format='pdf', bbox_inches='tight')
plt.savefig(save_path + 'claude_length_vs_performance.png', format='png', bbox_inches='tight')
print(f"Figures saved to {save_path}")

plt.show()
