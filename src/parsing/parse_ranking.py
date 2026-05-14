import json
import glob
import os
from typing import List, Dict

def parse_ranking_more(rank_string, doc_ids):
    rr = rank_string.split(">")
    
    scores = {}
    for i, p in enumerate(rr):
        for pidstring in p.split("="):
            pid = pidstring.strip().replace("[", "").replace("]", "")
            
            try: 
                pid = int(pid)
            except: 
                continue
            try:
                did = doc_ids[pid-1]
            except IndexError:
                continue
                
            if did in scores:
                return scores
            scores[did] = 1/(i+1)
        
    return scores

def process_files():
    # 匹配文件路径模式
    base_path = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/"
    pattern = base_path + "Rank-K-32B_dl20_rerank_depth_*_N_*_T_0.70_maxlen_*_random_*.jsonl"
    
    # 获取所有匹配的文件
    files = glob.glob(pattern)
    
    print(f"找到 {len(files)} 个匹配的文件")

    files = ['/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl19/Rank-K-32B_dl19_rerank_depth_20_N_1_T_0.50_maxlen_8000_random_17639191218648385.jsonl']
    
    for file_path in files:
        print(f"\n处理文件: {os.path.basename(file_path)}")
        
        # 读取文件内容
        processed_lines = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # 获取 doc_ids 和 model_responses
                    doc_ids = data.get('doc_ids', [])
                    model_responses = data.get('model_responses', [])
                    
                    # 处理每个 model_response
                    scores_list = []
                    for response in model_responses:
                        # 提取最后一句话（按换行符分割后的最后一行）
                        last_sentence = response.strip().split("\n")[-1]
                        
                        # 解析排名得到分数
                        scores = parse_ranking_more(last_sentence, doc_ids)
                        scores_list.append(scores)
                    
                    # 添加新字段
                    data['scores'] = scores_list
                    
                    # 添加到处理后的列表
                    processed_lines.append(json.dumps(data, ensure_ascii=False))
                    
                except Exception as e:
                    print(f"  错误 - 第 {line_num} 行: {e}")
                    # 保留原始行
                    processed_lines.append(line.strip())
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in processed_lines:
                f.write(line + '\n')
        
        print(f"  完成! 处理了 {len(processed_lines)} 行数据")

if __name__ == "__main__":
    process_files()
