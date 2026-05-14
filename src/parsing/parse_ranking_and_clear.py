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

def remove_redundancy_from_ranking(rank_string, doc_ids):
    """
    根据parse_ranking_more的逻辑进行去冗余
    当遇到重复的did时停止，返回去冗余后的ranking字符串
    """
    rr = rank_string.split(">")
    seen_dids = set()
    valid_parts = []
    
    for i, part in enumerate(rr):
        current_part_items = []
        should_stop = False
        
        items = part.split("=")
        for item in items:
            pid_str = item.strip().replace("[", "").replace("]", "")
            
            try:
                pid = int(pid_str)
            except:
                continue
            
            try:
                did = doc_ids[pid-1]
            except IndexError:
                continue
            
            if did in seen_dids:
                should_stop = True
                break
            
            seen_dids.add(did)
            current_part_items.append(item.strip())
        
        if current_part_items:
            valid_parts.append(" = ".join(current_part_items))
        
        if should_stop:
            break
    
    return " > ".join(valid_parts)

def process_files():
    # 匹配文件路径模式
    base_path = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/"
    pattern = base_path + "Rank-K-32B_dl20_rerank_depth_*_N_*_T_0.70_maxlen_*_random_*.jsonl"
    
    # 获取所有匹配的文件
    files = glob.glob(pattern)
    
    print(f"找到 {len(files)} 个匹配的文件")

    # files = ['/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl19/Rank-K-32B_dl19_rerank_depth_20_N_1_T_0.50_maxlen_8000_random_17639191218648385.jsonl']
    
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
                    cleared_responses = []
                    
                    for response in model_responses:
                        # 按换行符分割
                        lines = response.strip().split("\n")
                        
                        if lines:
                            # 前面所有字符串
                            prefix_lines = lines[:-1]
                            # 最后一句话
                            last_sentence = lines[-1]
                            
                            # 去冗余处理最后一句话
                            cleared_last_sentence = remove_redundancy_from_ranking(last_sentence, doc_ids)
                            
                            # 重新组合
                            if prefix_lines:
                                cleared_response = "\n".join(prefix_lines) + "\n" + cleared_last_sentence
                            else:
                                cleared_response = cleared_last_sentence
                            
                            cleared_responses.append(cleared_response)
                            
                            # 解析排名得到分数
                            scores = parse_ranking_more(last_sentence, doc_ids)
                            scores_list.append(scores)
                        else:
                            cleared_responses.append(response)
                            scores_list.append({})
                    
                    # 添加新字段
                    data['scores'] = scores_list
                    data['cleared_model_responses'] = cleared_responses
                    
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
