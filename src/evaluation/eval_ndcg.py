from .metrics import calculate_ndcg_at_k
import json
import math

def analyze_ndcg_mean(file_path: str, qrels_path: str, k: int = 10):
    """
    输出所有response对应NDCG@k均值（无长度区分，无绘图）
    
    Args:
        file_path: 数据文件路径
        qrels_path: qrels文件路径
        k: NDCG@k中的k值
    """
    # 加载qrels
    with open(qrels_path, 'r') as f:
        qrels = json.load(f)
    
    all_ndcg_scores = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                data = json.loads(line.strip())
                qid = str(data.get('qid'))
                doc_ids = data.get('doc_ids', [])
                # scores_list = [{doc_id:1/(rank+1) for rank,doc_id in enumerate(doc_ids)}]
                scores_list = data.get('scores', [])  # 每个response的打分列表

                for scores in scores_list:
                    if not scores:
                        continue
                    ndcg_k = calculate_ndcg_at_k(qid, scores, qrels, k=k)
                    all_ndcg_scores.append(ndcg_k)
            except Exception as e:
                print(f"处理数据时出错: {e}")
                continue

    total = len(all_ndcg_scores)
    if total > 0:
        mean_ndcg = sum(all_ndcg_scores) / total
        print(f"[文件: {file_path}]")
        print(f"  样本数: {total}")
        print(f"  平均NDCG@{k}: {mean_ndcg:.4f}")
    else:
        print(f"[文件: {file_path}] 没有有效NDCG分数.")

    return mean_ndcg if total > 0 else None

# 示例用法
if __name__ == "__main__":
    # qrels_path = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/data/dl19/qrels.json"
    # file_path = f"/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl19/Rank-K-32B_dl19_rerank_depth_20_N_1_T_0.50_maxlen_8000_random_17639191218648385.jsonl"
    # for k in [1,5,10]:
    #     analyze_ndcg_mean(file_path, qrels_path, k=k)

    qrels_path = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/data/dl20/qrels.json"
    file_path = f"/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20/Rank-K-32B_dl20_rerank_depth_20_N_1_T_0.50_maxlen_8000_random_17639191033250809.jsonl"
    for k in [1,5,10]:
        analyze_ndcg_mean(file_path, qrels_path, k=k)
