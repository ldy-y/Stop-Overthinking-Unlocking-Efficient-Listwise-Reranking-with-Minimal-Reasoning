import os
import random
import argparse
import json
from typing import List, Dict, Any, Iterable
from pathlib import Path
import time
import fcntl

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import vllm
from vllm import LLM, SamplingParams


# -----------------------------
# Utilities: seed & IO
# -----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def atomic_append_jsonl(filename: str, rows: Iterable[Dict[str, Any]]):
    """原子追加 JSONL"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def atomic_append_id(path: str, idx: int):
    """原子追加处理过的 ID"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(str(idx) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_processed_ids(path: str) -> set:
    """读取已处理的 ID"""
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip().isdigit()}


def time_seed_shuffle_indices(n: int) -> List[int]:
    """用当前时间+pid作为随机种子，避免不同实例顺序一致"""
    seed = int(time.time() * 1e7) ^ os.getpid()
    rnd = random.Random(seed)
    idxs = list(range(n))
    rnd.shuffle(idxs)
    return idxs


# -----------------------------
# Dataset loader: JSONL format
# -----------------------------
def load_rerank_data(args):
    """加载 JSON 格式的重排数据
    
    Expected format:
    [
        {"input": "query text", "hits": [
            {"content": "doc text", "qid": "xxx", "docid": "xxx", "rank": 1, "score": 14.85},
            ...
        ]},
        ...
    ]
    """
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 改为 json.load 而不是逐行读取
    
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in JSON file, got {type(data)}")
    
    print(f"[INFO] Total loaded: {len(data)} queries")
    
    # 应用 start/end 切片
    s = args.start if args.start is not None else 0
    e = args.end if args.end is not None else len(data)
    s = max(0, s)
    e = min(len(data), e)
    data = data[s:e]
    
    # 应用 eval_samples 限制
    if args.eval_samples is not None:
        n = min(args.eval_samples, len(data))
        data = data[:n]
    
    print(f"[INFO] After slicing: {len(data)} queries")
    return data


# -----------------------------
# Prompt building for reranking
# -----------------------------
def combine_passages(passages: List[str]) -> str:
    """格式化文档列表"""
    return "\n\n".join([
        f"[{i+1}] {p}" for i, p in enumerate(passages)
    ])


def create_rerank_prompt(query: str, docs: List[str]) -> str:
    """构建重排提示词（参考 rerank.py 的 rank_k_prompt）"""
    rank_k_prompt = """Determine a ranking of the passages based on how relevant they are to the query. 
If the query is a question, how relevant a passage is depends on how well it answers the question. 
If not, try analyze the intent of the query and assess how well each passage satisfy the intent. 
The query may have typos and passages may contain contradicting information. 
However, we do not get into fact-checking. We just rank the passages based on they relevancy to the query. 

Sort them from the most relevant to the least. 
Answer with the passage number using a format of `[3] > [2] > [4] = [1] > [5]`. 
Ties are acceptable if they are equally relevant. 
I need you to be accurate but overthinking it is unnecessary.
Output only the ordering without any other text.

Query: {query}

{docs}
"""
    return rank_k_prompt.format(
        query=query,
        docs=combine_passages(docs)
    )


# -----------------------------
# Parsing ranking output
# -----------------------------
def parse_ranking(rank_string: str, doc_ids: List[str]) -> Dict[str, float]:
    """解析模型输出的排序结果（参考 rerank.py 的 parse_ranking_more）"""
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
                did = doc_ids[pid - 1]
            except IndexError:
                continue
                
            if did in scores:
                return scores
            scores[did] = 1 / (i + 1)
        
    return scores


# -----------------------------
# Processed log path
# -----------------------------
def get_processed_log_path(args) -> str:
    model_name = Path(args.model_path).name
    data_name = Path(args.data_path).stem
    temp_str = f"{args.temperature:.2f}"
    fname = (
        f"processed_{model_name}_{data_name}_"
        f"depth_{args.rerank_depth}_window_{args.rerank_window}_"
        f"N_{args.num_samples}_T_{temp_str}_maxlen_{args.max_tokens}.log"
    )
    return os.path.join(args.output_dir, fname)


# -----------------------------
# Reranking with sliding window
# -----------------------------
def rerank_with_window(
    query: str,
    hits: List[Dict],
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer,
    args
) -> tuple[List[Dict], List[Dict]]:
    """使用滑动窗口进行重排（参考 rerank.py 的逻辑）"""
    
    if not hits:
        return []
    
    # 按原始 rank 排序（如果有的话，否则用 score）
    if 'rank' in hits[0]:
        hits_sorted = sorted(hits, key=lambda x: x['rank'])
    elif 'score' in hits[0]:
        hits_sorted = sorted(hits, key=lambda x: -x['score'])
    else:
        hits_sorted = hits
    
    # 只重排前 rerank_depth 个
    to_rerank = hits_sorted[:args.rerank_depth]
    rest = hits_sorted[args.rerank_depth:]
    
    # 提取 docid 和 content
    doc_ids = [hit['docid'] for hit in to_rerank]
    contents = [hit['content'] for hit in to_rerank]
    
    # 如果需要截断文档
    if args.truncate_doc_to is not None and tokenizer is not None:
        contents = [
            tokenizer.decode(tokenizer.encode(c, add_special_tokens=False)[:args.truncate_doc_to])
            for c in contents
        ]

    raw_outputs = []
    
    # 滑动窗口重排（从后往前）
    for rerank_end_idx in range(min(args.rerank_depth, len(doc_ids)), 0, -args.rerank_stride):
        rerank_start_idx = max(0, rerank_end_idx - args.rerank_window)
        
        window_doc_ids = doc_ids[rerank_start_idx:rerank_end_idx]
        window_contents = contents[rerank_start_idx:rerank_end_idx]
        
        # 构建 prompt
        prompt = create_rerank_prompt(query, window_contents)
        
        # 调用模型
        try:
            outputs = llm.generate([prompt], sampling_params)
            if not outputs or not outputs[0].outputs:
                ranking_text = ""
                raw_output = ""
            else:
                # 保存完整的原始输出
                raw_output = outputs[0].outputs[0].text.strip()
                # 取最后一行作为排序结果
                ranking_text = raw_output.split("\n")[-1]
                
            # 添加：记录原始输出
            raw_outputs.append({
                'window_start': rerank_start_idx,
                'window_end': rerank_end_idx,
                'prompt': prompt,  # 可选：也保存prompt
                'raw_response': raw_output,
                'parsed_ranking': ranking_text
            })
            
        except Exception as e:
            print(f"[WARNING] LLM generation failed: {e}, skip this window")
            ranking_text = ""
            raw_outputs.append({
                'window_start': rerank_start_idx,
                'window_end': rerank_end_idx,
                'error': str(e)
            })
        
        # 解析排序结果
        scores = parse_ranking(ranking_text, window_doc_ids)
        
        # 重新排序
        new_ranking = [
            doc_id for doc_id, score in sorted(scores.items(), key=lambda x: -x[1])
        ]
        
        # 补充未被排序的文档（保持原顺序）
        if len(new_ranking) < len(window_doc_ids):
            new_ranking += [d for d in window_doc_ids if d not in new_ranking]
        
        # 更新 doc_ids 和 contents
        doc_ids[rerank_start_idx:rerank_end_idx] = new_ranking
        # 同步更新 contents
        id_to_content = {wid: wc for wid, wc in zip(window_doc_ids, window_contents)}
        contents[rerank_start_idx:rerank_end_idx] = [id_to_content[did] for did in new_ranking]
        
        if rerank_start_idx == 0:
            break
    
    # 构建重排后的结果
    reranked_hits = []
    doc_id_to_hit = {hit['docid']: hit for hit in to_rerank}
    
    for new_rank, doc_id in enumerate(doc_ids, 1):
        hit = doc_id_to_hit[doc_id].copy()
        hit['rerank_score'] = 1 / new_rank
        hit['new_rank'] = new_rank
        reranked_hits.append(hit)
    
    # 添加未重排的文档
    for original_rank, hit in enumerate(rest, len(reranked_hits) + 1):
        hit_copy = hit.copy()
        hit_copy['rerank_score'] = 0.0
        hit_copy['new_rank'] = original_rank
        reranked_hits.append(hit_copy)
    
    return reranked_hits, raw_outputs


# -----------------------------
# Main reranking pipeline
# -----------------------------
def rerank_vllm(args, llm: LLM, tokenizer):
    print("Starting reranking with vLLM...")
    
    # 加载数据
    data = load_rerank_data(args)
    n_total = len(data)
    
    if n_total == 0:
        print("[WARNING] No data to process!")
        return
    
    # 时间标记
    time_based_seed = int(time.time() * 1e7) ^ os.getpid()
    
    # vLLM 采样参数（重排任务通常用低温度或贪婪）
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.num_samples,
        stop=args.stop if args.stop else [],
    )
    
    # 共享 processed 日志
    processed_log = get_processed_log_path(args)
    os.makedirs(os.path.dirname(processed_log), exist_ok=True)
    open(processed_log, "a").close()
    
    # 失败日志
    failed_log = processed_log.replace('.log', '_failed.log')
    open(failed_log, "a").close()
    
    # 输出文件
    model_name = Path(args.model_path).name
    data_name = Path(args.data_path).stem
    temp_str = f"{args.temperature:.2f}"
    out_name = (
        f"{model_name}_{data_name}_rerank_"
        f"depth_{args.rerank_depth}_window_{args.rerank_window}_"
        f"N_{args.num_samples}_T_{temp_str}_maxlen_{args.max_tokens}_random_{time_based_seed}.jsonl"
    )
    out_path = os.path.join(args.output_dir, out_name)
    
    # 打印信息
    initial_processed = load_processed_ids(processed_log)
    print(f"[INFO] Worker output -> {out_path}")
    print(f"[INFO] random_tag={time_based_seed}")
    print(f"[INFO] Shared processed log -> {processed_log} (current {len(initial_processed)} ids)")
    print(f"[INFO] Rerank parameters: depth={args.rerank_depth}, window={args.rerank_window}, stride={args.rerank_stride}")
    
    # 随机遍历顺序
    order = time_seed_shuffle_indices(n_total)
    
    processed_count = 0
    for pos, idx in enumerate(tqdm(order, desc="Reranking")):
        # 检查是否已处理
        processed = load_processed_ids(processed_log)
        if idx in processed:
            continue
        
        item = data[idx]
        query = item.get('input', '')
        hits = item.get('hits', [])
        
        if not query or not hits:
            print(f"[WARNING] idx={idx} has empty query or hits, skipping")
            atomic_append_id(processed_log, idx)
            continue
        
        try:
            # 执行重排 - 接收原始输出
            reranked_hits, raw_outputs = rerank_with_window( 
                query, hits, llm, sampling_params, tokenizer, args
            )
            
            # 构建输出
            result = {
                'idx': idx,
                'input': query,
                'qid': hits[0].get('qid', '') if hits else '',
                'original_hit_count': len(hits),
                'reranked_hits': reranked_hits,
                'raw_model_outputs': raw_outputs,
            }
            
            # 可选：保留原始 hits（如果需要对比）
            if args.keep_original:
                result['original_hits'] = hits
            
            # 写入结果
            atomic_append_jsonl(out_path, [result])
            processed_count += 1
            
            # 标记已处理
            atomic_append_id(processed_log, idx)
            
            if processed_count % 10 == 0:
                print(f"[PROGRESS] Processed {processed_count} queries", flush=True)
            
        except Exception as e:
            # 记录失败
            with open(failed_log, 'a') as f:
                f.write(f"{idx}\t{repr(e)}\n")
            print(f"[ERROR] idx={idx} reranking failed: {e}", flush=True)
            continue
    
    print(f"All done for this worker. Processed {processed_count} new queries.")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Reranking with vLLM - adapted from vllm_sample.py"
    )

    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the reranking model (e.g., hltcoe/Rank-K-32B)"
    )

    # Dataset
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to JSONL file with queries and hits"
    )

    # Ranging
    parser.add_argument("--start", type=int, default=None, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--eval_samples", type=int, default=None, help="Limit total samples")

    # Seed
    parser.add_argument("--seed", type=int, default=42)

    # Reranking parameters
    parser.add_argument(
        "--rerank_depth",
        type=int,
        default=100,
        help="Number of top documents to rerank"
    )
    parser.add_argument(
        "--rerank_window",
        type=int,
        default=20,
        help="Number of documents in each reranking window"
    )
    parser.add_argument(
        "--rerank_stride",
        type=int,
        default=10,
        help="Stride of sliding window"
    )
    parser.add_argument(
        "--truncate_doc_to",
        type=int,
        default=None,
        help="Truncate documents to N tokens"
    )

    # Generation
    parser.add_argument("--temperature", type=float, default=0.0, 
                       help="Temperature for generation (0.0 for greedy)")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Max tokens for reranking output")
    parser.add_argument("--stop", type=str, nargs="*", default=None,
                       help="Stop tokens")

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--keep_original",
        action="store_true",
        help="Keep original hits in output (for comparison)"
    )

    # vLLM runtime
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=32768)

    args = parser.parse_args()
    
    # Set seed
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer (for truncation if needed)
    tokenizer = None
    if args.truncate_doc_to is not None:
        print(f"[INFO] Loading tokenizer for document truncation...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )

    # Initialize vLLM
    print(f"[INFO] Initializing vLLM with {torch.cuda.device_count()} GPUs...")
    tp = torch.cuda.device_count()
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )

    # Run reranking
    rerank_vllm(args=args, llm=llm, tokenizer=tokenizer)


if __name__ == "__main__":
    main()


'''
python /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/vllm_sample_rerank.py \
  --model_path /mnt/ali-sh-1/usr/ningyikai/allmodels/Rank-K-32B \
  --data_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/old/data/dl20/dl20.json \
  --output_dir /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20 \
  --truncate_doc_to 450 \
  --rerank_depth 100 \
  --rerank_window 20 \
  --rerank_stride 10 \
  --temperature 0.5 \
  --num_samples 16 \
  --max_tokens 512 \
  --gpu_mem_util 0.9



python /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/vllm_sample_rerank.py \
  --model_path /mnt/ali-sh-1/usr/ningyikai/allmodels/Rank-K-32B \
  --data_path /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/old/data/dl20/dl20.json \
  --output_dir /mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/results/dl20 \
  --truncate_doc_to 450 \
  --rerank_depth 10 \
  --rerank_window 10 \
  --rerank_stride 10 \
  --temperature 0.5 \
  --num_samples 16 \
  --max_tokens 8000 \
  --gpu_mem_util 0.9

'''