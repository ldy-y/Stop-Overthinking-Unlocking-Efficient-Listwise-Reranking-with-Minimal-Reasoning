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
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer
import sys

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


def write_jsonl(filename: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(filename: str, rows: Iterable[Dict[str, Any]]):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def atomic_append_lines(path: str, lines: Iterable[str]):
    """对共享文件做原子追加（独占锁），不负责判重。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        for line in lines:
            if not line.endswith("\n"):
                line = line + "\n"
            f.write(line)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def atomic_append_id(path: str, idx: int):
    atomic_append_lines(path, [str(idx)])


def load_processed_ids(path: str) -> set:
    """读当前 processed 日志。不加锁，接受轻微竞态，只做 best-effort 全局视图。"""
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {int(line.strip()) for line in f if line.strip().isdigit()}


def time_seed_shuffle_indices(n: int) -> List[int]:
    """用当前时间+pid作为随机种子，避免不同实例顺序一致。"""
    seed = int(time.time() * 1e7) ^ os.getpid()
    rnd = random.Random(seed)
    idxs = list(range(n))
    rnd.shuffle(idxs)
    return idxs


# -----------------------------
# Dataset loader: must have problem/solution/answer
# -----------------------------
def load_hf_dataset(args):
    ds_obj = load_from_disk(args.dataset_path)

    if isinstance(ds_obj, DatasetDict):
        if args.split is None:
            raise ValueError(
                f"这个路径包含多个 split：{list(ds_obj.keys())}。请显式传 --split。"
            )
        if args.split not in ds_obj:
            raise ValueError(f"--split={args.split} 不存在，可选：{list(ds_obj.keys())}")
        ds = ds_obj[args.split]
    elif isinstance(ds_obj, Dataset):
        ds = ds_obj
        if args.split is not None:
            print(f"[警告] 数据不是 DatasetDict，忽略 --split={args.split}")
    else:
        raise ValueError("load_from_disk 返回的对象既不是 Dataset 也不是 DatasetDict。")

    cols = ds.column_names
    required = ["problem", "solution", "answer"]
    for k in required:
        if k not in cols:
            raise ValueError(f"数据缺少必需字段 '{k}'。现有列：{cols}")

    if args.id_field and args.id_field in cols:
        task_ids_all = [str(x) for x in ds[args.id_field]]
    else:
        task_ids_all = [str(i) for i in range(len(ds))]

    problems_all = ds["problem"]
    solutions_all = ds["solution"]
    answers_all = ds["answer"]

    s = args.start if args.start is not None else 0
    e = args.end if args.end is not None else len(problems_all)
    s = max(0, s)
    e = min(len(problems_all), e)

    task_ids = task_ids_all[s:e]
    problems = problems_all[s:e]
    solutions = solutions_all[s:e]
    answers = answers_all[s:e]

    if args.eval_samples is not None:
        n = min(args.eval_samples, len(problems))
        task_ids = task_ids[:n]
        problems = problems[:n]
        solutions = solutions[:n]
        answers = answers[:n]

    return task_ids, problems, solutions, answers


# -----------------------------
# Prompt building
# -----------------------------
def create_math_reasoning_prompt(system_prompt: str, problem_text: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Please reason step by step, and put your final answer within \\boxed{}."
                "Problem: " + str(problem_text)
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# -----------------------------
# Processed log path（共享）
# -----------------------------
def get_processed_log_path(args) -> str:
    # 区分模型/数据集/分片/采样次数/温度/上下文长（不含 seed）
    model_name = Path(args.model_path).name
    ds_base = Path(args.dataset_path).name
    dataset_name = f"{ds_base}_{args.split}" if args.split else ds_base
    temp_str = f"{args.temperature:.2f}"
    fname = (
        f"processed_{model_name}_{dataset_name}_N_{args.num_samples}_T_{temp_str}_L_{args.max_len}.log"
    )
    return os.path.join(args.output_dir, fname)


# -----------------------------
# Evaluation pipeline —— 单条即落盘
# -----------------------------
def evaluate_model_vllm(args, llm: LLM, tokenizer):
    print("Starting model evaluation with vLLM (math reasoning, no code)...")

    # 数据
    task_ids, problems, solutions, answers = load_hf_dataset(args)
    n_total = len(task_ids)

    # 时间相关随机 tag（仅用于区分输出文件）
    time_based_seed = int(time.time() * 1e7) ^ os.getpid()

    # 提前构建 prompts
    system_prompt = "You are a mathematics expert."
    prompts_all = [
        create_math_reasoning_prompt(system_prompt, prob, tokenizer)
        for prob in problems
    ]

    # vLLM 采样参数
    stop_tokens = args.stop if args.stop is not None else []
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_len,
        n=args.num_samples,
        stop=stop_tokens,
        extra_args={"moe_trace": True},  # <-- 开关
    )

    # 共享 processed 日志（touch）
    processed_log = get_processed_log_path(args)
    os.makedirs(os.path.dirname(processed_log), exist_ok=True)
    open(processed_log, "a").close()

    # 失败日志
    failed_log = os.path.join(
        args.output_dir,
        Path(processed_log).with_suffix("").name + "_failed.log",
    )
    open(failed_log, "a").close()

    # 本 worker 输出文件（带 time_based_seed）
    model_name = Path(args.model_path).name
    ds_base = Path(args.dataset_path).name
    dataset_name = f"{ds_base}_{args.split}" if args.split else ds_base
    temp_str = f"{args.temperature:.2f}"
    out_name = (
        f"{model_name}_{dataset_name}_N_{args.num_samples}_T_{temp_str}"
        f"_SEED_{args.seed}_L_{args.max_len}_random_{time_based_seed}.jsonl"
    )
    out_path = os.path.join(args.output_dir, out_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # 打印当前已处理数量
    initial_processed = load_processed_ids(processed_log)
    print(f"[INFO] Worker output -> {out_path}")
    print(f"[INFO] random_tag={time_based_seed}")
    print(
        f"[INFO] Shared processed log -> {processed_log} (current {len(initial_processed)} ids)"
    )

    # 随机遍历顺序（时间种子）
    order = time_seed_shuffle_indices(n_total)

    for pos, idx in enumerate(order):
        # 每次采样前，从全局 log 读取一次，确保尽量不重复
        processed = load_processed_ids(processed_log)
        if idx in processed:
            continue

        payload = {
            "task_id": task_ids[idx],
            "problem": problems[idx],
            "solution": solutions[idx],
            "answer": answers[idx],
        }
        prompt = prompts_all[idx]

        try:
            # 先采样，成功后再记 processed，保证“不中途挂就不会漏样本”
            outputs = llm.generate([prompt], sampling_params)
            if not outputs or not outputs[0].outputs:
                comps = [""]
            else:
                comps = [o.text.strip() for o in outputs[0].outputs]

            payload["completions"] = comps

            # 写结果
            append_jsonl(out_path, [payload])
            print(f"[WRITE] idx={idx} -> {out_path}", flush=True)

            # 成功后再 append 到 processed_log（加锁，避免文件损坏）
            atomic_append_id(processed_log, idx)

        except Exception as e:
            # 采样失败不写 processed，让其他 worker/下次重试有机会覆盖它
            atomic_append_lines(
                failed_log,
                [f"{idx}\t{repr(e)}"],
            )
            print(f"[ERROR] idx={idx} generation failed: {e}", flush=True)
            continue

    print("All done for this worker.")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/bd-1/usr/yuanpeiwen/allmodels/Qwen3-30B-A3B-Instruct-2507",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="（建议=1）单条调用；不影响本实现但过大无意义",
    )

    # Dataset
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/bd-1/usr/yuanpeiwen/wxl_file/MOE/dataset/AIME2024",
    )
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--id_field", type=str, default=None)

    # Ranging
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--eval_samples", type=int, default=None)

    # Seed
    parser.add_argument("--seed", type=int, default=42)

    # Generation
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_len", type=int, default=16384)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--stop", type=str, nargs="*", default=None)

    # Paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/bd-1/usr/yuanpeiwen/wxl_file/MOE/result",
    )

    # vLLM runtime
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=32768)

    args = parser.parse_args()
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # tokenizer 只用于 apply_chat_template
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # vLLM 实例
    tp = torch.cuda.device_count()
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )

    evaluate_model_vllm(args=args, llm=llm, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
