
# from metrics import calculate_ndcg_at_k

# import json
# import math
# from typing import Dict
# from collections import defaultdict
# import matplotlib.pyplot as plt
# from pathlib import Path
# from transformers import AutoTokenizer

# def analyze_ndcg_by_response_length(file_path: str, 
#                                   qrels_path: str, 
#                                   length_interval: int = 500, 
#                                   k: int = 10,
#                                   rerank_depth: int = 10,
#                                   save_plot: bool = True,
#                                   tokenizer_path: str = "/mnt/ali-sh-1/usr/ningyikai/allmodels/Rank-K-32B"):
#     """
#     分析不同response长度下的NDCG表现
    
#     Args:
#         file_path: merged数据文件路径
#         qrels_path: qrels文件路径
#         length_interval: 长度区间间隔，默认500（tokens）
#         k: NDCG@k中的k值，默认10
#         rerank_depth: rerank深度
#         save_plot: 是否保存图片，默认True
#         tokenizer_path: tokenizer路径
#     """
#     # 1. 加载tokenizer
#     print("Loading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#     print("Tokenizer loaded successfully!")
    
#     # 2. 加载qrels
#     with open(qrels_path, 'r') as f:
#         qrels = json.load(f)
    
#     # 存储不同长度区间的NDCG结果
#     length_buckets = defaultdict(list)  # {length_bucket: [ndcg_scores]}
    
#     # 读取数据并计算NDCG
#     print("Processing data...")
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line_idx, line in enumerate(f):
#             if not line.strip():
#                 continue
            
#             try:
#                 data = json.loads(line.strip())
#                 qid = str(data.get('qid'))
#                 doc_ids = data.get('doc_ids', [])
#                 scores_list = data.get('scores', [])
#                 model_responses = data.get('model_responses', [])
                
#                 # 处理每个response和对应的scores
#                 for i, (response, scores) in enumerate(zip(model_responses, scores_list)):
#                     if response is None or not scores:
#                         continue
                    
#                     # 使用tokenizer计算response长度（tokens数量）
#                     try:
#                         tokens = tokenizer.encode(response, add_special_tokens=False)
#                         response_length = len(tokens)
#                     except Exception as e:
#                         print(f"Tokenization error for line {line_idx}, response {i}: {e}")
#                         continue
                    
#                     # 确定长度区间 (length_interval*i到length_interval*(i+1))
#                     bucket = response_length // length_interval
#                     bucket_start = bucket * length_interval
#                     bucket_end = (bucket + 1) * length_interval
#                     bucket_name = f"{bucket_start}-{bucket_end}"

#                     # 计算NDCG@k
#                     ndcg_k = calculate_ndcg_at_k(qid, scores, qrels, k=k)
                    
#                     # 添加到对应的长度区间
#                     length_buckets[bucket_name].append(ndcg_k)
                
#             except Exception as e:
#                 print(f"处理数据时出错 (line {line_idx}): {e}")
#                 continue
    
#     print("Data processing completed!")
    
#     # 用于绘图的数据
#     plot_x = []  # 长度区间的中点
#     plot_y = []  # 对应的平均NDCG
    
#     for bucket_name in sorted(length_buckets.keys(), key=lambda x: int(x.split('-')[0])):
#         ndcg_scores = length_buckets[bucket_name]
#         avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
#         max_ndcg = max(ndcg_scores)
        
#         # 计算区间中点用于绘图
#         bucket_start, bucket_end = map(int, bucket_name.split('-'))
#         bucket_center = (bucket_start + bucket_end) // 2
#         plot_x.append(bucket_center)
#         plot_y.append(avg_ndcg)
    
#     # 4. 整体统计
#     all_ndcg_scores = []
#     for scores in length_buckets.values():
#         all_ndcg_scores.extend(scores)
    
#     # 5. 绘制并保存图片
#     if save_plot and plot_x and plot_y:
#         plt.figure(figsize=(12, 6))
#         plt.plot(plot_x, plot_y, marker='o', linewidth=2, markersize=6)
#         plt.xlabel('Response Length (tokens)', fontsize=12)  # 修改标签为tokens
#         plt.ylabel(f'Average NDCG@{k}', fontsize=12)
#         plt.title(f'NDCG@{k} vs Response Length in Tokens (interval={length_interval})', fontsize=14)  # 修改标题
#         plt.grid(True, alpha=0.3)
        
#         # 添加数据点标签
#         for i, (x, y) in enumerate(zip(plot_x, plot_y)):
#             bucket_name = sorted(length_buckets.keys(), key=lambda x: int(x.split('-')[0]))[i]
#             sample_count = len(length_buckets[bucket_name])
#             plt.annotate(f'{y:.3f}\n(n={sample_count})', 
#                         (x, y), 
#                         textcoords="offset points", 
#                         xytext=(0,10), 
#                         ha='center',
#                         fontsize=9)
        
#         plt.tight_layout()
        
#         # 保存图片到file_path同一目录
#         file_dir = Path(file_path).parent
#         plot_filename = f"merged_Rank-K-32B_dl20_rerank_depth_{rerank_depth}_ndcg_at_{k}_vs_response_tokens_interval_{length_interval}.png"
#         plot_path = file_dir / plot_filename
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         print(f"\n图片已保存至: {plot_path}")
#         plt.show()
    
#     return length_buckets


# if __name__ == "__main__":
#     qrels_path = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/data/dl20/qrels.json"
#     tokenizer_path = "/mnt/ali-sh-1/usr/ningyikai/allmodels/Rank-K-32B"
    
#     for rerank_depth in [10,20,30,40]:
#         file_path = f"/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/merged_Rank-K-32B_dl20_rerank_depth_{rerank_depth}_T_0.70.jsonl"
        
#         # 由于使用tokens，可能需要调整长度区间
#         # 一般tokens数量比字符数量少，可以使用较小的interval
#         analyze_ndcg_by_response_length(
#             file_path=file_path, 
#             qrels_path=qrels_path, 
#             length_interval=500, 
#             k=10, 
#             rerank_depth=rerank_depth,
#             tokenizer_path=tokenizer_path
#         )


from .metrics import calculate_ndcg_at_k

import json
import math
from typing import Dict
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer

def analyze_ndcg_by_response_length(file_path: str, 
                                  qrels_path: str, 
                                  length_interval: int = 500, 
                                  k: int = 10,
                                  rerank_depth: int = 10,
                                  save_plot: bool = True,
                                  tokenizer_path: str = "/mnt/ali-sh-1/usr/ningyikai/allmodels/Rank-K-32B",
                                  min_samples: int = 20):
    """
    分析不同response长度下的NDCG表现
    
    Args:
        file_path: merged数据文件路径
        qrels_path: qrels文件路径
        length_interval: 长度区间间隔，默认500（tokens）
        k: NDCG@k中的k值，默认10
        rerank_depth: rerank深度
        save_plot: 是否保存图片，默认True
        tokenizer_path: tokenizer路径
        min_samples: 每个区间最小样本数量，少于此数量会合并到下一个区间
    """
    # 1. 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("Tokenizer loaded successfully!")
    
    # 2. 加载qrels
    with open(qrels_path, 'r') as f:
        qrels = json.load(f)
    
    # 存储不同长度区间的NDCG结果
    length_buckets = defaultdict(list)  # {length_bucket: [ndcg_scores]}
    
    # 读取数据并计算NDCG
    print("Processing data...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line.strip())
                qid = str(data.get('qid'))
                doc_ids = data.get('doc_ids', [])
                scores_list = data.get('scores', [])
                model_responses = data.get('model_responses', [])
                
                # 处理每个response和对应的scores
                for i, (response, scores) in enumerate(zip(model_responses, scores_list)):
                    if response is None or not scores:
                        continue
                    
                    # 使用tokenizer计算response长度（tokens数量）
                    try:
                        tokens = tokenizer.encode(response, add_special_tokens=False)
                        response_length = len(tokens)
                    except Exception as e:
                        print(f"Tokenization error for line {line_idx}, response {i}: {e}")
                        continue
                    
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
                print(f"处理数据时出错 (line {line_idx}): {e}")
                continue
    
    print("Data processing completed!")
    
    # 3. 合并样本数量少于min_samples的区间
    print(f"Merging buckets with fewer than {min_samples} samples...")
    
    # 按区间起始位置排序
    sorted_bucket_names = sorted(length_buckets.keys(), key=lambda x: int(x.split('-')[0]))
    
    merged_buckets = defaultdict(list)
    
    i = 0
    while i < len(sorted_bucket_names):
        current_bucket = sorted_bucket_names[i]
        current_scores = length_buckets[current_bucket]
        
        # 如果当前区间样本数量足够，直接保留
        if len(current_scores) >= min_samples:
            merged_buckets[current_bucket] = current_scores
            i += 1
        else:
            # 如果样本数量不足，尝试与后续区间合并
            combined_scores = current_scores.copy()
            combined_bucket_start = int(current_bucket.split('-')[0])
            
            j = i + 1
            # 持续合并直到样本数量达到要求或没有更多区间
            while j < len(sorted_bucket_names) and len(combined_scores) < min_samples:
                next_bucket = sorted_bucket_names[j]
                next_scores = length_buckets[next_bucket]
                combined_scores.extend(next_scores)
                j += 1
            
            # 创建合并后的区间名称
            if j <= len(sorted_bucket_names):
                if j == len(sorted_bucket_names):
                    # 如果已经到最后了，使用最后一个区间的结束位置
                    last_bucket = sorted_bucket_names[j-1]
                    combined_bucket_end = int(last_bucket.split('-')[1])
                else:
                    # 使用下一个未处理区间的起始位置
                    combined_bucket_end = int(sorted_bucket_names[j-1].split('-')[1])
            else:
                combined_bucket_end = int(current_bucket.split('-')[1])
            
            combined_bucket_name = f"{combined_bucket_start}-{combined_bucket_end}"
            merged_buckets[combined_bucket_name] = combined_scores
            
            print(f"Merged buckets {current_bucket} to {sorted_bucket_names[min(j-1, len(sorted_bucket_names)-1)]} "
                  f"into {combined_bucket_name} (n={len(combined_scores)})")
            
            i = j
    
    # 更新length_buckets为合并后的结果
    length_buckets = merged_buckets
    
    # 4. 用于绘图的数据
    plot_x = []  # 长度区间的中点
    plot_y = []  # 对应的平均NDCG
    
    print("\nFinal bucket statistics:")
    for bucket_name in sorted(length_buckets.keys(), key=lambda x: int(x.split('-')[0])):
        ndcg_scores = length_buckets[bucket_name]
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
        max_ndcg = max(ndcg_scores)
        
        print(f"Bucket {bucket_name}: {len(ndcg_scores)} samples, avg NDCG@{k}: {avg_ndcg:.4f}")
        
        # 计算区间中点用于绘图
        bucket_start, bucket_end = map(int, bucket_name.split('-'))
        bucket_center = (bucket_start + bucket_end) // 2
        plot_x.append(bucket_center)
        plot_y.append(avg_ndcg)
    
    # 5. 整体统计
    all_ndcg_scores = []
    for scores in length_buckets.values():
        all_ndcg_scores.extend(scores)
    
    print(f"\nOverall statistics:")
    print(f"Total samples: {len(all_ndcg_scores)}")
    print(f"Overall average NDCG@{k}: {sum(all_ndcg_scores)/len(all_ndcg_scores):.4f}")
    
    # 6. 绘制并保存图片
    if save_plot and plot_x and plot_y:
        plt.figure(figsize=(12, 6))
        plt.plot(plot_x, plot_y, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Response Length (tokens)', fontsize=12)
        plt.ylabel(f'Average NDCG@{k}', fontsize=12)
        plt.title(f'NDCG@{k} vs Response Length in Tokens (interval={length_interval}, min_samples={min_samples})', fontsize=14)
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
        plot_filename = f"merged_Rank-K-32B_dl20_rerank_depth_{rerank_depth}_ndcg_at_{k}_vs_response_tokens_interval_{length_interval}_minsamples_{min_samples}.png"
        plot_path = file_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n图片已保存至: {plot_path}")
        plt.show()
    
    return length_buckets


if __name__ == "__main__":
    qrels_path = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/data/dl20/qrels.json"
    tokenizer_path = "/mnt/ali-sh-1/usr/ningyikai/allmodels/Rank-K-32B"
    
    for rerank_depth in [20,10, 30,40]:
        file_path = f"/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/merged_Rank-K-32B_dl20_rerank_depth_{rerank_depth}_T_0.70.jsonl"
        
        # 由于使用tokens，可能需要调整长度区间
        # 一般tokens数量比字符数量少，可以使用较小的interval
        analyze_ndcg_by_response_length(
            file_path=file_path, 
            qrels_path=qrels_path, 
            length_interval=100, 
            k=10, 
            rerank_depth=rerank_depth,
            tokenizer_path=tokenizer_path,
            min_samples=50  # 设置最小样本数量
        )
