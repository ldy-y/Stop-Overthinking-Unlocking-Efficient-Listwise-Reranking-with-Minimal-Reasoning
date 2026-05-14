#!/usr/bin/env python3
import os
import json
import argparse
from collections import OrderedDict
from typing import List, Dict

def find_result_files(result_dir: str, model: str, dataset: str, N: int, T: float, L: int) -> List[str]:
    """
    按采样命名规则筛选：
    {model}_{dataset}_N_{N}_T_{T:.2f}_SEED_*_L_{L}_random_*.jsonl
    并按 mtime 升序排序（先写入的文件优先）
    """
    t_str = f"{T:.2f}"
    prefix = f"{model}_{dataset}_N_{N}_T_{t_str}_SEED_"
    mid = f"_L_{L}_random_"
    files = []
    try:
        for fn in os.listdir(result_dir):
            if not fn.endswith(".jsonl"):
                continue
            if not fn.startswith(prefix):
                continue
            if mid not in fn:
                continue
            files.append(os.path.join(result_dir, fn))
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] result_dir 不存在：{result_dir}")

    files.sort(key=lambda p: os.path.getmtime(p))
    return files

def merge_jsonl_first_seen(files: List[str], cap_n: int) -> List[Dict]:
    """
    以 task_id 为主键合并；completions 去重且先到先得；每个任务最多 cap_n 条。
    其它字段（problem/solution/answer）保留首次出现版本。
    """
    merged: Dict[str, Dict] = OrderedDict()

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue

                tid = str(item.get("task_id", "")).strip()
                if not tid:
                    continue

                # 拿到 completions（兼容老字段）
                comps = item.get("completions")
                if not isinstance(comps, list):
                    comps = item.get("model_outputs", [])
                if not isinstance(comps, list):
                    comps = []

                if tid not in merged:
                    merged[tid] = {
                        "task_id": tid,
                        "problem": item.get("problem"),
                        "solution": item.get("solution"),
                        "answer": item.get("answer"),
                        "completions": []
                    }

                bucket = merged[tid]["completions"]
                if len(bucket) >= cap_n:
                    continue

                # 去重保序（基于完整字符串）
                seen = set(bucket)
                for c in comps:
                    if len(bucket) >= cap_n:
                        break
                    c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
                    if c in seen:
                        continue
                    bucket.append(c)
                    seen.add(c)

    return list(merged.values())

def main():
    ap = argparse.ArgumentParser(description="Merge vLLM sampling results from multiple machines.")
    ap.add_argument("--result_dir", type=str, required=True, help="各机输出 jsonl 所在目录")
    ap.add_argument("--model", type=str, required=True, help="模型名（文件名前缀的一部分）例如 Qwen3-30B-A3B")
    ap.add_argument("--dataset", type=str, required=True, help="数据集名（文件名前缀的一部分）例如 MATH500 或 MATH500_test")
    ap.add_argument("--N", type=int, default=64, help="采样次数 N（文件名里的 N_{N}），默认 64")
    ap.add_argument("--T", type=float, required=True, help="温度 T（文件名里的 T_{T:.2f}，两位小数匹配）")
    ap.add_argument("--L", type=int, required=True, help="max_len（文件名里的 L_{L}）")
    ap.add_argument("--output", type=str, default="", help="合并输出路径；默认自动命名到 result_dir 下")
    args = ap.parse_args()

    files = find_result_files(args.result_dir, args.model, args.dataset, args.N, args.T, args.L)
    if not files:
        raise SystemExit(
            f"[ERROR] 没找到文件：{args.result_dir} 下符合 "
            f"{args.model}_{args.dataset}_N_{args.N}_T_{args.T:.2f}_SEED_*_L_{args.L}_random_*.jsonl 的结果。"
        )

    print(f"[INFO] 待合并文件数：{len(files)}")
    for p in files:
        print(f"  - {os.path.basename(p)}")

    merged = merge_jsonl_first_seen(files, cap_n=args.N)

    if not args.output:
        out_name = f"{args.model}_{args.dataset}_N_{args.N}_T_{args.T:.2f}_L_{args.L}_merged.jsonl"
        args.output = os.path.join(args.result_dir, out_name)

    with open(args.output, "w", encoding="utf-8") as f:
        for obj in merged:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] 合并完成：样本数={len(merged)} → {args.output}")

if __name__ == "__main__":
    main()
