from .metrics import calculate_ndcg_at_k
import json
import os
import math

def load_qrels(qrels_path: str):
    """
    支持两种 qrels 格式：
    1) JSON: {qid: {docid: gain}}
    2) TXT: 每行 4 列: qid  [ignored]  docid  gain
    """
    ext = os.path.splitext(qrels_path)[1].lower()
    
    if ext == '.json':
        with open(qrels_path, 'r', encoding='utf-8') as f:
            qrels = json.load(f)
        # 确保 key 都是字符串
        qrels_str = {}
        for qid, docs in qrels.items():
            qid_str = str(qid)
            qrels_str[qid_str] = {str(docid): int(gain) for docid, gain in docs.items()}
        return qrels_str
    
    # 默认按 txt 4 列格式读取
    qrels = {}
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                # 跳过格式不对的行
                continue
            qid, _, docid, gain = parts[0], parts[1], parts[2], parts[3]
            try:
                gain_int = int(gain)
            except ValueError:
                # 非法 gain，当作 0
                gain_int = 0
            qid = str(qid)
            docid = str(docid)
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = gain_int
    return qrels


def analyze_dl19_ndcg(file_path: str, qrels_path: str, k: int = 10):
    """
    针对 dl19.json 格式 (hits 列表) 计算 NDCG@k
    支持 qrels_path 为 JSON 或 4 列 txt（qid _ docid gain）
    """
    # 加载 qrels
    qrels = load_qrels(qrels_path)
    
    all_ndcg_scores = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data_list = json.load(f)
        except json.JSONDecodeError:
            print("JSON解析失败，请检查预测文件是否为标准JSON格式")
            return 0.0

        for item in data_list:
            try:
                hits = item.get('hits', [])
                if not hits:
                    continue

                # 获取 QID
                qid = str(hits[0].get('qid'))

                # 提取 DocIDs
                doc_ids = [str(hit.get('docid')) for hit in hits]

                # 构建打分字典（位置越靠前，分数越高）
                scores = {doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(doc_ids)}

                # 计算 NDCG@k
                ndcg_k = calculate_ndcg_at_k(qid, scores, qrels, k=k)
                all_ndcg_scores.append(ndcg_k)

            except Exception as e:
                import traceback
                print(f"处理单条数据时出错 (QID: {qid if 'qid' in locals() else 'Unknown'}): {e}")
                # traceback.print_exc()
                continue

    total = len(all_ndcg_scores)
    if total > 0:
        mean_ndcg = sum(all_ndcg_scores) / total
        print(f"[结果] NDCG@{k} 均值: {mean_ndcg:.4f} (样本数: {total})")
    else:
        print(f"[结果] 没有计算出有效的 NDCG 分数。")
        mean_ndcg = 0.0

    return mean_ndcg


import argparse

def main():
    parser = argparse.ArgumentParser(description="计算 DL19 数据集的 NDCG 指标")
    
    parser.add_argument(
        "--qrels_path", 
        type=str, 
        required=True,
        help="qrels.json 文件路径"
    )
    parser.add_argument(
        "--file_path", 
        type=str, 
        required=True,
        help="检索结果文件路径"
    )
    parser.add_argument(
        "--k_values", 
        type=int, 
        nargs="+", 
        # default=[1, 5, 10],
        default=[10],
        help="NDCG@k 的 k 值列表，默认为 1 5 10"
    )
    
    args = parser.parse_args()

    print(args.file_path)
    
    print("-" * 30)
    for k in args.k_values:
        analyze_dl19_ndcg(args.file_path, args.qrels_path, k=k)

# import os

# def main():
#     # 根据你的实际 Bright 子数据集修改这里
#     datasets = [
#         "aops",
#         "biology",
#         "earth-science",
#         "economics",
#         "leetcode",
#         "pony",
#         "psychology",
#         "robotics",
#         "stackoverflow",
#         "sustainable-living",
#         "theoremqa-questions",
#         "theoremqa-theorems",
#     ]

#     base_qrels = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/Bright/RAW/qrel"
#     base_runs  = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/Bright/SPLADE_V3"

#     k_values = [10]   # 如需 [1,5,10] 就改成 [1, 5, 10]

#     for dataset in datasets:
#         qrels_path = os.path.join(base_qrels, f"{dataset.replace('-','_')}.txt")
#         file_path  = os.path.join(base_runs,  f"retrieve_results_bright-{dataset}_top100.json")

#         if not os.path.exists(qrels_path):
#             print(f"[Bright-{dataset}] qrels 不存在，跳过: {qrels_path}")
#             continue
#         if not os.path.exists(file_path):
#             print(f"[Bright-{dataset}] 结果文件不存在，跳过: {file_path}")
#             continue

#         print("=" * 80)
#         print(f"[Bright-{dataset}]")
#         print(f"qrels_path: {qrels_path}")
#         print(f"file_path:  {file_path}")
#         print("-" * 30)

#         for k in k_values:
#             analyze_dl19_ndcg(file_path, qrels_path, k=k)


if __name__ == "__main__":
    main()

'''
# 基本用法
python /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Scripts/OverThink/cal_ndcg.py \
    --qrels_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/DL19/RAW/2019qrels-pass.txt \
    --file_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/dl19/dl19.json


------------------------ Dl19 ------------------------

python /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Scripts/OverThink/cal_ndcg.py \
    --qrels_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/DL19/RAW/2019qrels-pass.txt \
    --file_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/DL19/BM25/retrieve_results_dl19_top100.json

python /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Scripts/OverThink/cal_ndcg.py \
    --qrels_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/DL19/RAW/2019qrels-pass.txt \
    --file_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/DL19/SPLADE_V3/retrieve_results_dl19_top100.json
------------------------ Dl20 ------------------------
python /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Scripts/OverThink/cal_ndcg.py \
    --qrels_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/DL20/RAW/2020qrels-pass.txt \
    --file_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/DL20/BM25/retrieve_results_dl20_top100.json

python /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Scripts/OverThink/cal_ndcg.py \
    --qrels_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/DL20/RAW/2020qrels-pass.txt \
    --file_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/DL20/SPLADE_V3/retrieve_results_dl20_top100.json


------------------------ fas ------------------------
python /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Scripts/OverThink/cal_ndcg.py \
    --qrels_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/NeuCLIR/RAW/qrels/all_qrels.txt \
    --file_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/NeuCLIR/BM25/retrieve_results_neuclir-fas_top100.json

------------------------ rus ------------------------
python /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Scripts/OverThink/cal_ndcg.py \
    --qrels_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/NeuCLIR/RAW/qrels/all_qrels.txt \
    --file_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/NeuCLIR/BM25/retrieve_results_neuclir-rus_top100.json
------------------------ zho ------------------------
python /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Scripts/OverThink/cal_ndcg.py \
    --qrels_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/NeuCLIR/RAW/qrels/all_qrels.txt \
    --file_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/NeuCLIR/BM25/retrieve_results_neuclir-zho_top100.json



------------------------ bright ------------------------
python /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Scripts/OverThink/cal_ndcg.py \
    --qrels_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/Bright/RAW/qrel/biology.txt \
    --file_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/Bright/BM25/retrieve_results_bright-biology_top100.json

python /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Scripts/OverThink/cal_ndcg.py \
    --qrels_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/Bright/RAW/qrel/biology.txt \
    --file_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/Bright/SPLADE_V3/retrieve_results_bright-biology_top100.json


python /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Scripts/OverThink/cal_ndcg.py \
    --qrels_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/Bright/RAW/qrel/earth_science.txt \
    --file_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/Bright/BM25/retrieve_results_bright-earth-science_top100.json

python /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Scripts/OverThink/cal_ndcg.py \
    --qrels_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/Bright/RAW/qrel/earth_science.txt \
    --file_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rerank/Data/TestSet/Bright/SPLADE_V3/retrieve_results_bright-earth-science_top100.json


'''