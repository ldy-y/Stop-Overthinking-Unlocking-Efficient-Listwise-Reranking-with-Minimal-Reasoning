from .metrics import calculate_ndcg_at_k

import json
import math
from typing import Dict
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_ndcg_by_response_length(file_path: str, 
                                  qrels_path: str, 
                                  length_interval: int = 500, 
                                  k: int = 10,
                                  rerank_depth: int = 10,
                                  save_plot: bool = True):
    """
    分析不同response长度下的NDCG表现
    
    Args:
        file_path: merged数据文件路径
        qrels_path: qrels文件路径
        length_interval: 长度区间间隔，默认500
        k: NDCG@k中的k值，默认10
        save_plot: 是否保存图片，默认True
    """
    # 1. 加载qrels
    with open(qrels_path, 'r') as f:
        qrels = json.load(f)
    
    # 存储不同长度区间的NDCG结果
    length_buckets = defaultdict(list)  # {length_bucket: [ndcg_scores]}
    
    # 读取数据并计算NDCG
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line.strip())
                qid = str(data.get('qid'))
                doc_ids = data.get('doc_ids', [])
                scores_list = data.get('scores', [])
                model_responses = data.get('model_responses', [])  # 注意这里是model_response不是model_responses
                
                # 处理每个response和对应的scores
                for i, (response, scores) in enumerate(zip(model_responses, scores_list)):
                    if response is None or not scores:
                        continue
                    
                    # 计算response长度
                    response_length = len(response)
                    
                    # 确定长度区间 (length_interval*i到length_interval*(i+1))
                    bucket = response_length // length_interval
                    bucket_start = bucket * length_interval
                    bucket_end = (bucket + 1) * length_interval
                    bucket_name = f"{bucket_start}-{bucket_end}"

                    # 计算NDCG@k
                    ndcg_k = calculate_ndcg_at_k(qid, scores, qrels, k=k)
                    
                    # 添加到对应的长度区间
                    length_buckets[bucket_name].append(ndcg_k)
                
            except Exception as e:
                print(f"处理数据时出错: {e}")
                continue
    
    # # 3. 计算并输出结果
    # print(f"不同Response长度区间的NDCG@{k}表现:")
    # print("=" * 60)

    # 用于绘图的数据
    plot_x = []  # 长度区间的中点
    plot_y = []  # 对应的平均NDCG
    
    for bucket_name in sorted(length_buckets.keys(), key=lambda x: int(x.split('-')[0])):
        ndcg_scores = length_buckets[bucket_name]
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
        max_ndcg = max(ndcg_scores)
        
        # 计算区间中点用于绘图
        bucket_start, bucket_end = map(int, bucket_name.split('-'))
        bucket_center = (bucket_start + bucket_end) // 2
        plot_x.append(bucket_center)
        plot_y.append(avg_ndcg)
        # plot_y.append(max_ndcg)
        
        # print(f"长度区间 {bucket_name}:")
        # print(f"  样本数量: {len(ndcg_scores)}")
        # print(f"  平均NDCG@{k}: {avg_ndcg:.4f}")
        # print(f"  最大NDCG@{k}: {max(ndcg_scores):.4f}")
        # print(f"  最小NDCG@{k}: {min(ndcg_scores):.4f}")
        # print()
    
    # 4. 整体统计
    all_ndcg_scores = []
    for scores in length_buckets.values():
        all_ndcg_scores.extend(scores)
    
    # if all_ndcg_scores:
    #     print("整体统计:")
    #     print(f"  总样本数: {len(all_ndcg_scores)}")
    #     print(f"  整体平均NDCG@{k}: {sum(all_ndcg_scores)/len(all_ndcg_scores):.4f}")
    #     print(f"  整体最大NDCG@{k}: {max(all_ndcg_scores):.4f}")
    #     print(f"  整体最小NDCG@{k}: {min(all_ndcg_scores):.4f}")
    
    # 5. 绘制并保存图片
    if save_plot and plot_x and plot_y:
        plt.figure(figsize=(12, 6))
        plt.plot(plot_x, plot_y, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Response Length (characters)', fontsize=12)
        plt.ylabel(f'Average NDCG@{k}', fontsize=12)
        # plt.ylabel(f'Max NDCG@{k}', fontsize=12)
        plt.title(f'NDCG@{k} vs Response Length (interval={length_interval})', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加数据点标签
        for i, (x, y) in enumerate(zip(plot_x, plot_y)):
            bucket_name = sorted(length_buckets.keys(), key=lambda x: int(x.split('-')[0]))[i]
            sample_count = len(length_buckets[bucket_name])
            plt.annotate(f'{y:.3f}\n(n={sample_count})', 
                        (x, y), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=9)
        
        plt.tight_layout()
        
        # 保存图片到file_path同一目录
        file_dir = Path(file_path).parent
        plot_filename = f"merged_Rank-K-32B_dl20_rerank_depth_{rerank_depth}_ndcg_at_{k}_vs_response_length_interval_{length_interval}.png"
        # plot_filename = f"merged_Rank-K-32B_dl20_rerank_depth_{rerank_depth}_max_ndcg_at_{k}_vs_response_length_interval_{length_interval}.png"
        plot_path = file_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n图片已保存至: {plot_path}")
        plt.show()
    
    return length_buckets


if __name__ == "__main__":
    qrels_path = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/data/dl20/qrels.json"
    for rerank_depth in [10,20,30,40]:
        file_path = f"/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/merged_Rank-K-32B_dl20_rerank_depth_{rerank_depth}_T_0.70.jsonl"
        
        # 或者自定义参数
        # for k in [1,5,10]:
        #     for length_interval in [2000, 5000, 10000]:
        #         analyze_ndcg_by_response_length(file_path, qrels_path, length_interval=length_interval, k=k, rerank_depth=rerank_depth)

        analyze_ndcg_by_response_length(file_path, qrels_path, length_interval=2000, k=10, rerank_depth=rerank_depth)
