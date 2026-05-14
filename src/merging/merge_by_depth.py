import json
import glob
import re
from collections import defaultdict
from pathlib import Path


def extract_params_from_filename(filename):
    """从文件名中提取参数"""
    pattern = r'rerank_depth_(\d+)_N_(\d+)_T_0\.70_maxlen_(\d+)_random_(\d+)'
    match = re.search(pattern, filename)
    if match:
        rerank_depth = int(match.group(1))
        N = int(match.group(2))
        maxlen = int(match.group(3))
        random_seed = match.group(4)
        return rerank_depth, N, maxlen, random_seed
    return None


def merge_scores_by_depth():
    base_path = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/"
    pattern = base_path + "Rank-K-32B_dl20_rerank_depth_*_N_*_T_0.70_maxlen_*_random_*.jsonl"
    
    # 获取所有匹配的文件
    files = glob.glob(pattern)
    print(f"找到 {len(files)} 个匹配的文件\n")
    
    # 存储数据: {rerank_depth: {idx: data}}
    # 使用idx作为唯一标识（相同input的idx应该相同）
    depth_data = defaultdict(dict)
    
    # 统计信息
    stats = defaultdict(lambda: {'total_scores': 0, 'filtered_scores': 0})
    
    # 处理每个文件
    for file_path in files:
        params = extract_params_from_filename(file_path)
        if not params:
            print(f"警告: 无法解析文件名 {file_path}")
            continue
        
        rerank_depth, N, maxlen, random_seed = params
        print(f"处理: depth={rerank_depth}, N={N}, maxlen={maxlen}, seed={random_seed}")
        
        count = 0
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line.strip())
                    
                    idx = data.get('idx')
                    scores_list = data.get('scores', [])
                    model_responses_list = data.get('model_responses', [])
                    
                    stats[rerank_depth]['total_scores'] += len(scores_list)
                    
                    # 过滤：只保留长度等于rerank_depth的scores，同时过滤对应的model_response
                    filtered_scores = []
                    filtered_model_responses = []
                    
                    for i, scores in enumerate(scores_list):
                        if len(scores) == rerank_depth:
                            filtered_scores.append(scores)
                            # 如果model_responses_list存在且索引有效，则添加对应的response
                            if i < len(model_responses_list):
                                filtered_model_responses.append(model_responses_list[i])
                            else:
                                filtered_model_responses.append(None)  # 或者空字符串 ""
                    
                    stats[rerank_depth]['filtered_scores'] += len(filtered_scores)
                    
                    # 如果这是第一次遇到这个idx
                    if idx not in depth_data[rerank_depth]:
                        depth_data[rerank_depth][idx] = {
                            'idx': idx,
                            'input': data.get('input'),
                            'qid': data.get('qid', idx),  # 如果没有qid字段，使用idx
                            'rerank_depth': rerank_depth,
                            'doc_ids': data.get('doc_ids'),
                            'scores': [],
                            'model_responses': []
                        }
                    
                    # 合并scores和model_response
                    depth_data[rerank_depth][idx]['scores'].extend(filtered_scores)
                    depth_data[rerank_depth][idx]['model_responses'].extend(filtered_model_responses)
                    count += 1
                    
                except Exception as e:
                    print(f"  错误: {e}")
                    continue
        
        print(f"  处理了 {count} 条记录")

    # 输出合并后的文件
    print("\n" + "=" * 80)
    print("写入合并后的文件：")
    print("=" * 80)
    
    for rerank_depth in sorted(depth_data.keys()):
        output_file = base_path + f"merged_Rank-K-32B_dl20_rerank_depth_{rerank_depth}_T_0.70.jsonl"
        
        # 按idx排序写入
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx in sorted(depth_data[rerank_depth].keys()):
                data = depth_data[rerank_depth][idx]
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        num_queries = len(depth_data[rerank_depth])
        total_scores = sum(len(data['scores']) for data in depth_data[rerank_depth].values())
        avg_scores = total_scores / num_queries if num_queries > 0 else 0
        
        print(f"\nDepth {rerank_depth}:")
        print(f"  Queries: {num_queries}")
        print(f"  Total valid scores: {total_scores}")
        print(f"  Avg scores per query: {avg_scores:.2f}")
        print(f"  Total scores before filter: {stats[rerank_depth]['total_scores']}")
        print(f"  Filtered scores: {stats[rerank_depth]['filtered_scores']}")
        print(f"  Filter rate: {stats[rerank_depth]['filtered_scores']/stats[rerank_depth]['total_scores']*100:.2f}%")
        print(f"  Output: {output_file}")
    
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)


if __name__ == "__main__":
    merge_scores_by_depth()
