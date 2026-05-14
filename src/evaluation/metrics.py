import math
from typing import Dict

# def calculate_ndcg_at_k(qid, predicted_scores, qrels, k=10):
#     query_qrels = qrels.get(qid, {})
#     if not query_qrels:
#         return 0.0

#     # 按分数降序排列预测文档
#     ranked_doc_ids = sorted(predicted_scores.keys(), key=lambda x: predicted_scores[x], reverse=True)
#     ranked_doc_ids = ranked_doc_ids[:k]

#     # 计算 DCG
#     dcg = 0.0
#     for i, doc_id in enumerate(ranked_doc_ids):
#         raw_rel = query_qrels.get(doc_id, 0)
#         try:
#             rel = int(raw_rel)  # <--- 关键修复：强制转int
#         except ValueError:
#             rel = 0
        
#         if rel > 0:
#             dcg += rel / math.log2(i + 2) # log2(rank+1), rank=i+1 -> log2(i+2)

#     # 计算 IDCG
#     try:
#         all_rels = [int(r) for r in query_qrels.values()] # <--- 关键修复
#     except ValueError:
#         all_rels = [0]
    
#     true_relevances = sorted(all_rels, reverse=True)[:k]
    
#     idcg = 0.0
#     for i, rel in enumerate(true_relevances):
#         if rel > 0:
#             idcg += rel / math.log2(i + 2)

#     if idcg == 0:
#         return 0.0

#     return dcg / idcg


import math

def calculate_ndcg_at_k(qid, predicted_scores, qrels, k=10):
    # 原始 qrels：qid -> {doc_id: raw_rel}
    raw_query_qrels = qrels.get(qid, {})
    if not raw_query_qrels:
        return 0.0

    # 1. 建立 doc_id -> rel(int) 的字典，并过滤掉 rel < 0 的文档
    #    注意：这里不再直接用 raw_query_qrels，而是用处理后的 query_qrels
    query_qrels = {}
    for doc_id, raw_rel in raw_query_qrels.items():
        try:
            rel = int(raw_rel)
        except ValueError:
            rel = 0
        # 过滤掉 rel < 0 的文档（这些文档完全不参与评测）
        if rel >= 0:
            query_qrels[doc_id] = rel

    # 如果没有任何 (rel >= 0) 的文档，则直接返回 0
    if not query_qrels:
        return 0.0

    # 2. 按分数降序排列预测文档，并截断到前 k 个
    ranked_doc_ids = sorted(
        predicted_scores.keys(),
        key=lambda x: predicted_scores[x],
        reverse=True
    )
    ranked_doc_ids = ranked_doc_ids[:k]

    # 3. 计算 DCG：未在 qrels 中的 doc_id -> get(..., 0) 当作 rel=0
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids):
        rel = query_qrels.get(doc_id, 0)  # 未标注的当作 0
        if rel > 0:
            dcg += rel / math.log2(i + 2)  # rank=i+1 -> log2(i+2)

    # 4. 计算 IDCG：只用 query_qrels 中的 rel>=0，再取前 k 个
    all_rels = list(query_qrels.values())
    true_relevances = sorted(all_rels, reverse=True)[:k]

    idcg = 0.0
    for i, rel in enumerate(true_relevances):
        if rel > 0:
            idcg += rel / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg



# def calculate_ndcg_at_k(qid, predicted_scores, qrels, k=10):
#     query_qrels = qrels.get(qid, {})
#     if not query_qrels:
#         return 0.0

#     # 按分数降序排列预测文档
#     ranked_doc_ids = sorted(predicted_scores.keys(), key=lambda x: predicted_scores[x], reverse=True)
#     ranked_doc_ids = ranked_doc_ids[:k]

#     # 计算 DCG（使用 trec_eval 公式）
#     dcg = 0.0
#     for i, doc_id in enumerate(ranked_doc_ids):
#         raw_rel = query_qrels.get(doc_id, 0)
#         try:
#             rel = int(raw_rel)
#         except ValueError:
#             rel = 0
        
#         if rel > 0:
#             gain = (2 ** rel - 1)  # <--- 关键修改
#             dcg += gain / math.log2(i + 2)

#     # 计算 IDCG
#     try:
#         all_rels = [int(r) for r in query_qrels.values()]
#     except ValueError:
#         all_rels = [0]
    
#     true_relevances = sorted(all_rels, reverse=True)[:k]
    
#     idcg = 0.0
#     for i, rel in enumerate(true_relevances):
#         if rel > 0:
#             gain = (2 ** rel - 1)  # <--- 关键修改
#             idcg += gain / math.log2(i + 2)

#     if idcg == 0:
#         return 0.0

#     return dcg / idcg


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import json
    
    # 1. 加载qrels
    qrels_path = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/data/dl20/qrels.json"
    with open(qrels_path, 'r') as f:
        qrels = json.load(f)
    
    # 2. 示例scores（从parse_ranking_more得到）
    qid = "23849"
    scores = {"8816622": 1.0, "8816620": 0.5, "1034183": 0.3}
    
    # 3. 计算NDCG@k
    ndcg_1 = calculate_ndcg_at_k(qid, scores, qrels, k=1)
    ndcg_5 = calculate_ndcg_at_k(qid, scores, qrels, k=5)
    ndcg_10 = calculate_ndcg_at_k(qid, scores, qrels, k=10)
    
    print(f"Query {qid}:")
    print(f"  NDCG@1:  {ndcg_1:.4f}")
    print(f"  NDCG@5:  {ndcg_5:.4f}")
    print(f"  NDCG@10: {ndcg_10:.4f}")
