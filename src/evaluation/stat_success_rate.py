
import json
import glob
import re
from collections import defaultdict
import numpy as np

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

def analyze_query(scores_list, target_depth):
    """分析单个query的成功情况"""
    if not scores_list:
        return {
            'success_rate': 0.0,
            'pass_at_1': 0,
            'success_count': 0,
            'total_count': 0
        }
    
    success_count = 0
    for scores_dict in scores_list:
        if len(scores_dict) == target_depth:
            success_count += 1
    
    return {
        'success_rate': success_count / len(scores_list),
        'pass_at_1': 1 if success_count > 0 else 0,  # 至少成功一次
        'success_count': success_count,
        'total_count': len(scores_list)
    }

def main():
    base_path = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/"
    pattern = base_path + "Rank-K-32B_dl20_rerank_depth_*_N_*_T_0.70_maxlen_*_random_*.jsonl"
    
    # 获取所有匹配的文件
    files = glob.glob(pattern)
    print(f"找到 {len(files)} 个匹配的文件\n")
    
    # 存储数据: {(rerank_depth, maxlen, N): {'success_rates': [], 'pass_at_1': [], ...}}
    stats = defaultdict(lambda: {
        'success_rates': [],
        'pass_at_1': [],
        'success_counts': [],
        'total_counts': [],
        'seeds': []
    })
    
    # 处理每个文件
    for file_path in files:
        params = extract_params_from_filename(file_path)
        if not params:
            print(f"警告: 无法解析文件名 {file_path}")
            continue
        
        rerank_depth, N, maxlen, random_seed = params
        key = (rerank_depth, maxlen, N)  # 三维key
        
        print(f"处理: depth={rerank_depth}, maxlen={maxlen}, N={N}, seed={random_seed}")
        
        # 读取文件中的每一行（每个query）
        query_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line.strip())
                    scores_list = data.get('scores', [])
                    
                    # 分析这个query
                    result = analyze_query(scores_list, rerank_depth)
                    
                    stats[key]['success_rates'].append(result['success_rate'])
                    stats[key]['pass_at_1'].append(result['pass_at_1'])
                    stats[key]['success_counts'].append(result['success_count'])
                    stats[key]['total_counts'].append(result['total_count'])
                    
                    query_count += 1
                    
                except Exception as e:
                    print(f"  错误: {e}")
        
        stats[key]['seeds'].append(random_seed)
        print(f"  处理了 {query_count} 个 queries\n")
    
    # 计算并输出统计结果
    print("=" * 120)
    print("统计结果（按 Depth × MaxLen × N 分组）：")
    print("=" * 120)
    print(f"{'Depth':<8} {'MaxLen':<10} {'N':<8} {'Queries':<10} {'Avg Success':<15} "
          f"{'Pass@1':<12} {'Min':<8} {'Max':<8} {'Median':<10} {'Seeds':<10}")
    print("-" * 120)
    
    # 按 rerank_depth, maxlen, N 排序
    results = []
    for key in sorted(stats.keys()):
        rerank_depth, maxlen, N = key
        data = stats[key]
        
        success_rates = data['success_rates']
        pass_at_1_list = data['pass_at_1']
        
        avg_success_rate = np.mean(success_rates)
        pass_at_1_rate = np.mean(pass_at_1_list)  # pass@1的比率
        min_success = np.min(success_rates)
        max_success = np.max(success_rates)
        median_success = np.median(success_rates)
        num_seeds = len(data['seeds'])
        
        print(f"{rerank_depth:<8} {maxlen:<10} {N:<8} {len(success_rates):<10} "
              f"{avg_success_rate:<15.4f} {pass_at_1_rate:<12.4f} "
              f"{min_success:<8.4f} {max_success:<8.4f} {median_success:<10.4f} {num_seeds:<10}")
        
        results.append({
            'depth': rerank_depth,
            'maxlen': maxlen,
            'N': N,
            'num_queries': len(success_rates),
            'avg_success': avg_success_rate,
            'pass_at_1': pass_at_1_rate,
            'min': min_success,
            'max': max_success,
            'median': median_success,
            'std': np.std(success_rates),
            'num_seeds': num_seeds
        })
    
    # 详细的分布统计
    print("\n" + "=" * 120)
    print("详细统计（包含标准差和分位数）：")
    print("=" * 120)
    print(f"{'Depth':<8} {'MaxLen':<10} {'N':<8} {'Avg':<12} {'Pass@1':<12} "
          f"{'Std':<10} {'25%':<10} {'50%':<10} {'75%':<10}")
    print("-" * 120)
    
    for key in sorted(stats.keys()):
        rerank_depth, maxlen, N = key
        success_rates = stats[key]['success_rates']
        pass_at_1_rate = np.mean(stats[key]['pass_at_1'])
        
        avg = np.mean(success_rates)
        std = np.std(success_rates)
        q25 = np.percentile(success_rates, 25)
        q50 = np.percentile(success_rates, 50)
        q75 = np.percentile(success_rates, 75)
        
        print(f"{rerank_depth:<8} {maxlen:<10} {N:<8} {avg:<12.4f} {pass_at_1_rate:<12.4f} "
              f"{std:<10.4f} {q25:<10.4f} {q50:<10.4f} {q75:<10.4f}")

if __name__ == "__main__":
    main()
