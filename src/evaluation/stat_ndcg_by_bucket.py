from .metrics import calculate_ndcg_at_k

import json
import math
from typing import Dict
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def analyze_ndcg_by_bucket(file_path: str, 
                          qrels_path: str, 
                          k: int = 10,
                          rerank_depth: int = 10,
                          maxl: int = 8192,
                          save_plot: bool = True,
                          bucket_type: str = "equal_width"):
    """
    分析不同桶的NDCG表现
    
    Args:
        file_path: 包含桶信息的数据文件路径
        qrels_path: qrels文件路径
        k: NDCG@k中的k值，默认10
        rerank_depth: rerank深度
        save_plot: 是否保存图片，默认True
        bucket_type: 桶类型，"equal_width" 或 "equal_freq"
    """
    
    # 选择要使用的桶字段
    if bucket_type == "equal_width":
        bucket_field = "equal_width_bucket_ids"
        title_suffix = "Equal-Width Buckets"
    elif bucket_type == "equal_freq":
        bucket_field = "equal_freq_bucket_ids"
        title_suffix = "Equal-Frequency Buckets"
    else:
        raise ValueError("bucket_type must be 'equal_width' or 'equal_freq'")
    
    # 加载qrels
    with open(qrels_path, 'r') as f:
        qrels = json.load(f)
    
    # 存储不同桶的NDCG结果
    bucket_ndcg = defaultdict(list)  # {bucket_id: [ndcg_scores]}
    bucket_token_counts = defaultdict(list)  # {bucket_id: [token_counts]} 用于分析
    
    # 读取数据并计算NDCG
    print(f"Processing data for {bucket_type} buckets...")
    total_samples = 0
    skipped_samples = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line.strip())
                qid = str(data.get('qid'))
                doc_ids = data.get('doc_ids', [])
                scores_list = data.get('scores', [])
                bucket_ids = data.get(bucket_field, [])
                token_counts = data.get('cleared_model_response_tokens_num', [])
                
                # 检查数据完整性
                if not all(len(lst) == len(scores_list) for lst in [bucket_ids, token_counts]):
                    print(f"Warning: Length mismatch for qid {qid}, skipping...")
                    continue
                
                # 处理每个response和对应的scores、bucket_id
                for i, (scores, bucket_id, token_count) in enumerate(zip(scores_list, bucket_ids, token_counts)):
                    if scores is None or bucket_id is None:
                        skipped_samples += 1
                        continue
                    
                    total_samples += 1
                    
                    # 计算NDCG@k
                    ndcg_k = calculate_ndcg_at_k(qid, scores, qrels, k=k)
                    
                    # 添加到对应的桶
                    bucket_ndcg[bucket_id].append(ndcg_k)
                    bucket_token_counts[bucket_id].append(token_count)
                
            except Exception as e:
                print(f"处理数据时出错 (line {line_idx}): {e}")
                continue
    
    print(f"Data processing completed!")
    print(f"Total samples processed: {total_samples}")
    print(f"Skipped samples: {skipped_samples}")
    
    # 统计信息
    print(f"\nBucket statistics for {bucket_type}:")
    bucket_stats = {}
    
    for bucket_id in sorted(bucket_ndcg.keys()):
        ndcg_scores = bucket_ndcg[bucket_id]
        token_counts_for_bucket = bucket_token_counts[bucket_id]
        
        if len(ndcg_scores) == 0:
            continue
            
        avg_ndcg = np.mean(ndcg_scores)
        std_ndcg = np.std(ndcg_scores)
        min_ndcg = np.min(ndcg_scores)
        max_ndcg = np.max(ndcg_scores)
        
        avg_tokens = np.mean(token_counts_for_bucket)
        min_tokens = np.min(token_counts_for_bucket)
        max_tokens = np.max(token_counts_for_bucket)
        
        bucket_stats[bucket_id] = {
            'count': len(ndcg_scores),
            'avg_ndcg': avg_ndcg,
            'std_ndcg': std_ndcg,
            'min_ndcg': min_ndcg,
            'max_ndcg': max_ndcg,
            'avg_tokens': avg_tokens,
            'min_tokens': min_tokens,
            'max_tokens': max_tokens
        }
        
        print(f"Bucket {bucket_id}:")
        print(f"  Samples: {len(ndcg_scores)}")
        print(f"  NDCG@{k}: {avg_ndcg:.4f} ± {std_ndcg:.4f} (min: {min_ndcg:.4f}, max: {max_ndcg:.4f})")
        print(f"  Token range: {min_tokens:.0f} - {max_tokens:.0f} (avg: {avg_tokens:.1f})")
        print()
    
    # 整体统计
    all_ndcg_scores = []
    for scores in bucket_ndcg.values():
        all_ndcg_scores.extend(scores)
    
    if all_ndcg_scores:
        print(f"Overall statistics:")
        print(f"Total samples with valid NDCG: {len(all_ndcg_scores)}")
        print(f"Overall average NDCG@{k}: {np.mean(all_ndcg_scores):.4f} ± {np.std(all_ndcg_scores):.4f}")
    
    # 绘制图表
    if save_plot and bucket_stats:
        # 准备绘图数据
        bucket_ids = sorted(bucket_stats.keys())
        avg_ndcgs = [bucket_stats[bid]['avg_ndcg'] for bid in bucket_ids]
        std_ndcgs = [bucket_stats[bid]['std_ndcg'] for bid in bucket_ids]
        sample_counts = [bucket_stats[bid]['count'] for bid in bucket_ids]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 上图：NDCG@k vs 桶号
        bars = ax1.bar(bucket_ids, avg_ndcgs, yerr=std_ndcgs, capsize=5, 
                       alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_xlabel('Bucket ID', fontsize=12)
        ax1.set_ylabel(f'Average NDCG@{k}', fontsize=12)
        ax1.set_title(f'NDCG@{k} by {title_suffix} (rerank_depth={rerank_depth})', fontsize=14)
        ax1.set_xticks(bucket_ids)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bucket_id, avg_ndcg, count) in enumerate(zip(bucket_ids, avg_ndcgs, sample_counts)):
            ax1.text(bucket_id, avg_ndcg + std_ndcgs[i] + 0.005, 
                    f'{avg_ndcg:.3f}\n(n={count})', 
                    ha='center', va='bottom', fontsize=10)
        
        # 下图：样本数量分布
        bars2 = ax2.bar(bucket_ids, sample_counts, alpha=0.7, 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_xlabel('Bucket ID', fontsize=12)
        ax2.set_ylabel('Sample Count', fontsize=12)
        ax2.set_title('Sample Distribution by Bucket', fontsize=14)
        ax2.set_xticks(bucket_ids)
        ax2.grid(True, alpha=0.3)
        
        # 添加样本数量标签
        for bucket_id, count in zip(bucket_ids, sample_counts):
            ax2.text(bucket_id, count + max(sample_counts) * 0.01, 
                    f'{count}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图片
        file_dir = Path(file_path).parent
        plot_filename = f"ndcg_at_{k}_by_{bucket_type}_4buckets_rerank_depth_{rerank_depth}_maxlen_{maxl}.png"
        plot_path = file_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n图片已保存至: {plot_path}")
        plt.show()
        
        # 额外保存详细的token分布图
        plt.figure(figsize=(12, 6))
        
        # 准备token范围数据
        token_ranges = []
        for bucket_id in bucket_ids:
            min_tokens = bucket_stats[bucket_id]['min_tokens']
            max_tokens = bucket_stats[bucket_id]['max_tokens']
            avg_tokens = bucket_stats[bucket_id]['avg_tokens']
            token_ranges.append((min_tokens, max_tokens, avg_tokens))
        
        # 绘制token范围
        for i, (bucket_id, (min_tokens, max_tokens, avg_tokens)) in enumerate(zip(bucket_ids, token_ranges)):
            plt.errorbar(bucket_id, avg_tokens, 
                        yerr=[[avg_tokens - min_tokens], [max_tokens - avg_tokens]], 
                        fmt='o', capsize=8, capthick=2, markersize=8,
                        label=f'Bucket {bucket_id}')
            plt.text(bucket_id, max_tokens + (max([tr[1] for tr in token_ranges]) * 0.02), 
                    f'{avg_tokens:.0f}', ha='center', fontsize=10)
        
        plt.xlabel('Bucket ID', fontsize=12)
        plt.ylabel('Token Count', fontsize=12)
        plt.title(f'Token Distribution by {title_suffix}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(bucket_ids)
        
        # 保存token分布图
        token_plot_filename = f"token_distribution_by_{bucket_type}_4buckets_rerank_depth_{rerank_depth}_maxlen_{maxl}.png"
        token_plot_path = file_dir / token_plot_filename
        plt.savefig(token_plot_path, dpi=300, bbox_inches='tight')
        print(f"Token分布图已保存至: {token_plot_path}")
        plt.show()
    
    return bucket_stats


if __name__ == "__main__":
    qrels_path = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/data/dl20/qrels.json"
    
    for rerank_depth in [20]: #, 10, 30, 40]:
        for maxl in [8192, 16384]:
            file_path = f"/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/merged_Rank-K-32B_dl20_rerank_depth_{rerank_depth}_N_64_T_0.70_maxlen_{maxl}_with_4buckets.jsonl"
            
            print(f"\n{'='*60}")
            print(f"Processing rerank_depth = {rerank_depth}")
            print(f"{'='*60}")
            
            # 分析等宽分桶
            print(f"\nAnalyzing Equal-Width Buckets...")
            equal_width_stats = analyze_ndcg_by_bucket(
                file_path=file_path, 
                qrels_path=qrels_path, 
                k=10, 
                rerank_depth=rerank_depth,
                maxl = maxl,
                bucket_type="equal_width"
            )
            
            # 分析等频分桶
            print(f"\nAnalyzing Equal-Frequency Buckets...")
            equal_freq_stats = analyze_ndcg_by_bucket(
                file_path=file_path, 
                qrels_path=qrels_path, 
                k=10, 
                rerank_depth=rerank_depth,
                maxl = maxl,
                bucket_type="equal_freq"
            )
            
            print(f"\nCompleted processing for rerank_depth = {rerank_depth}")
