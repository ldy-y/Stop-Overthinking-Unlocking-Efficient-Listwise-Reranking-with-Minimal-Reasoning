import json
import pandas as pd
import numpy as np

def read_jsonl_file(file_path):
    """读取JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def assign_buckets_per_query(data, num_buckets=4):
    """为每个query下的采样结果分配桶号（等值和等频两种方式）
    
    Args:
        data: 输入数据
        num_buckets: 分桶数量，默认为4
    """
    
    for item in data:
        qid = item['qid']
        token_nums = item['cleared_model_response_tokens_num']
        
        # 如果token_nums为空，跳过
        if not token_nums:
            item['equal_width_bucket_ids'] = []
            item['equal_freq_bucket_ids'] = []
            continue
            
        # 将token数量转换为numpy数组便于处理
        token_array = np.array(token_nums)
        min_tokens = np.min(token_array)
        max_tokens = np.max(token_array)
        
        # === 等值分桶（Equal Width Binning） ===
        try:
            if min_tokens == max_tokens:
                # 所有值相同，全部分配到桶0
                equal_width_bucket_ids = [0] * len(token_nums)
                bin_edges = [min_tokens, max_tokens + 1]
            else:
                # 计算等宽区间的边界
                bin_width = (max_tokens - min_tokens) / num_buckets
                bin_edges = []
                for i in range(num_buckets + 1):
                    if i == num_buckets:
                        bin_edges.append(max_tokens + 1)  # +1 确保最大值被包含
                    else:
                        bin_edges.append(min_tokens + i * bin_width)
                
                # 分配桶号 (0, 1, 2, ..., num_buckets-1)
                equal_width_bucket_ids = []
                for token_num in token_nums:
                    bucket_id = 0
                    for i in range(1, len(bin_edges)):
                        if token_num < bin_edges[i]:
                            bucket_id = i - 1
                            break
                    equal_width_bucket_ids.append(bucket_id)
                    
        except Exception as e:
            print(f"Warning: Cannot create equal-width buckets for qid {qid}, assigning all to bucket 0. Error: {e}")
            equal_width_bucket_ids = [0] * len(token_nums)
            bin_edges = [min_tokens, max_tokens + 1]
        
        # === 等频分桶（Equal Frequency Binning） ===
        try:
            if len(set(token_nums)) <= 1:
                # 所有值相同或只有一个唯一值，全部分配到桶0
                equal_freq_bucket_ids = [0] * len(token_nums)
                percentiles = [min_tokens] * (num_buckets - 1)
            else:
                # 计算分位数
                percentile_points = []
                for i in range(1, num_buckets):
                    percentile_points.append(i * 100.0 / num_buckets)
                
                percentiles = np.percentile(token_array, percentile_points)
                
                # 分配桶号 (0, 1, 2, ..., num_buckets-1)
                equal_freq_bucket_ids = []
                for token_num in token_nums:
                    bucket_id = 0
                    for i, threshold in enumerate(percentiles):
                        if token_num <= threshold:
                            bucket_id = i
                            break
                        else:
                            bucket_id = i + 1
                    equal_freq_bucket_ids.append(bucket_id)
                    
        except Exception as e:
            print(f"Warning: Cannot create equal-freq buckets for qid {qid}, assigning all to bucket 0. Error: {e}")
            equal_freq_bucket_ids = [0] * len(token_nums)
            percentiles = [min_tokens] * (num_buckets - 1)
        
        # 添加桶号列表到原数据
        item['equal_width_bucket_ids'] = equal_width_bucket_ids
        item['equal_freq_bucket_ids'] = equal_freq_bucket_ids
        
        # 打印每个query的分桶信息
        print(f"\nQID {qid}: Token range [{min_tokens}, {max_tokens}], Total samples: {len(token_nums)}")
        
        # 等宽分桶信息
        ew_bucket_counts = np.bincount(equal_width_bucket_ids, minlength=num_buckets)
        print(f"  Equal-Width Buckets ({num_buckets} buckets):")
        print(f"    Bin edges: {[f'{edge:.1f}' for edge in bin_edges]}")
        print(f"    Distribution: {ew_bucket_counts}")
        
        # 等频分桶信息
        ef_bucket_counts = np.bincount(equal_freq_bucket_ids, minlength=num_buckets)
        print(f"  Equal-Freq Buckets ({num_buckets} buckets):")
        print(f"    Percentiles: {[f'{p:.1f}' for p in percentiles]}")
        print(f"    Distribution: {ef_bucket_counts}")
    
    return data

def save_jsonl_file(data, output_path):
    """保存处理后的数据到JSONL文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 主处理流程
def main(num_buckets=4):
    """
    主处理函数
    
    Args:
        num_buckets: 分桶数量，默认为4
    """
    # 文件路径
    for maxl in [8192,16384]:
        input_file = f"/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/merged_Rank-K-32B_dl20_rerank_depth_20_N_64_T_0.70_maxlen_{maxl}.jsonl"
        output_file = input_file.replace('.jsonl', f'_with_{num_buckets}buckets.jsonl')
        
        # 读取数据
        print(f"正在读取文件... (将分为 {num_buckets} 个桶)")
        data = read_jsonl_file(input_file)
        print(f"成功读取 {len(data)} 条记录")
        
        # 查看数据结构示例
        if data:
            print("\n数据结构示例:")
            sample = data[0]
            for key, value in sample.items():
                if isinstance(value, list):
                    print(f"  {key}: list of length {len(value)}")
                    if value and key == 'cleared_model_response_tokens_num':
                        print(f"    示例值: {value[:5]}..." if len(value) > 5 else f"    值: {value}")
                else:
                    print(f"  {key}: {value}")
        
        # 为每个query分配桶号
        print(f"\n正在为每个query分配桶号... (分为 {num_buckets} 个桶)")
        processed_data = assign_buckets_per_query(data, num_buckets=num_buckets)
        
        # 保存结果
        print(f"\n正在保存结果到: {output_file}")
        save_jsonl_file(processed_data, output_file)
        
        # 统计信息
        print("\n" + "="*50)
        print(f"总体分桶统计信息 ({num_buckets} 个桶)")
        print("="*50)
        
        total_samples = 0
        total_ew_buckets = [0] * num_buckets  # equal width
        total_ef_buckets = [0] * num_buckets  # equal freq
        
        for item in processed_data:
            if item.get('equal_width_bucket_ids'):
                sample_count = len(item['equal_width_bucket_ids'])
                total_samples += sample_count
                
                # 统计等宽分桶
                ew_bucket_counts = np.bincount(item['equal_width_bucket_ids'], minlength=num_buckets)
                for i in range(num_buckets):
                    total_ew_buckets[i] += ew_bucket_counts[i]
                
                # 统计等频分桶
                ef_bucket_counts = np.bincount(item['equal_freq_bucket_ids'], minlength=num_buckets)
                for i in range(num_buckets):
                    total_ef_buckets[i] += ef_bucket_counts[i]
        
        print(f"总采样数: {total_samples}")
        print(f"\n等宽分桶分布:")
        for i in range(num_buckets):
            percentage = (total_ew_buckets[i] / total_samples * 100) if total_samples > 0 else 0
            print(f"  Bucket {i}: {total_ew_buckets[i]} ({percentage:.1f}%)")
        
        print(f"\n等频分桶分布:")
        for i in range(num_buckets):
            percentage = (total_ef_buckets[i] / total_samples * 100) if total_samples > 0 else 0
            print(f"  Bucket {i}: {total_ef_buckets[i]} ({percentage:.1f}%)")
        
        # 验证数据完整性
        print("\n" + "="*50)
        print("数据完整性验证")
        print("="*50)
        
        for item in processed_data:
            scores_len = len(item.get('scores', []))
            responses_len = len(item.get('model_responses', []))
            cleared_responses_len = len(item.get('cleared_model_responses', []))
            token_nums_len = len(item.get('cleared_model_response_tokens_num', []))
            ew_bucket_ids_len = len(item.get('equal_width_bucket_ids', []))
            ef_bucket_ids_len = len(item.get('equal_freq_bucket_ids', []))
            
            expected_len = scores_len
            lengths = [responses_len, cleared_responses_len, token_nums_len, ew_bucket_ids_len, ef_bucket_ids_len]
            
            if not all(l == expected_len for l in lengths):
                print(f"Warning: Length mismatch for qid {item.get('qid')}: "
                    f"expected={expected_len}, "
                    f"responses={responses_len}, cleared_responses={cleared_responses_len}, "
                    f"tokens={token_nums_len}, ew_buckets={ew_bucket_ids_len}, ef_buckets={ef_bucket_ids_len}")
            else:
                print(f"✓ QID {item.get('qid')}: All fields aligned with length {expected_len}")

        print(f"\n处理完成！结果已保存到: {output_file}")

if __name__ == "__main__":
    main(num_buckets=4)
    main(num_buckets=8)
