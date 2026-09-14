"""8-GPU evaluation script for Qwen3.5 on the AIME26 dataset.

Each worker owns one GPU, loads its own model replica, and evaluates a
disjoint shard of the test set.  Partial results are written to SHARD_DIR and
merged into RESULT_PATH by the parent process.
"""

import json
import os
import re
import traceback
from pathlib import Path

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MAX_NEW_TOKENS, TEMPERATURE, ENABLE_THINKING = 8192, 0.85, True
TOP_P, TOP_K, REPETITION_PENALTY = 0.95, 20, 1.05
MODEL_PATH = "/mnt/data/user/zhang_jingdong/models/Qwen3.5-2B"
DATASET_NAME = "math-ai/aime26"
HF_CACHE = "/mnt/data/user/zhang_jingdong/hf_cache"
RESULT_PATH = "/mnt/data/user/zhang_jingdong/NLP_test/eval_result.json"
SHARD_DIR = "/mnt/data/user/zhang_jingdong/NLP_test/sub_logs"
GPU_COUNT = 8


def _field(row, names, default=None):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def extract_answer(text):
    """Extract an integer answer, preferring boxed/final-answer forms."""
    boxed = re.findall(r"\\boxed\\s*\\{\\s*(-?\\d+)\\s*\\}", text)
    if boxed:
        return boxed[-1]
    marked = re.findall(r"(?:final answer|answer)\\s*[:：]\\s*(-?\\d+)", text, re.I)
    if marked:
        return marked[-1]
    nums = re.findall(r"-?\\d+", text)
    return nums[-1] if nums else None


def worker(rank, indices, rows):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = torch.device("cuda:0")
    shard_path = Path(SHARD_DIR) / f"shard_{rank}.json"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, cache_dir=HF_CACHE, dtype=torch.bfloat16,
            device_map={"": 0}, trust_remote_code=True,
        ).eval()
        results = []
        for idx in indices:
            row = rows[idx]
            problem = _field(row, ("problem", "question", "prompt", "input"), "")
            reference = _field(row, ("answer", "solution", "target"))
            messages = [{"role": "user", "content": str(problem) +
                        "\\nPlease solve this problem and give the final integer answer."}]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=ENABLE_THINKING,
                )
            except (TypeError, ValueError):
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                    top_p=TOP_P, top_k=TOP_K, repetition_penalty=REPETITION_PENALTY,
                    do_sample=True, pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            pred = extract_answer(generated)
            ref = extract_answer(str(reference)) if reference is not None else None
            results.append({"index": idx, "problem": problem, "response": generated,
                            "prediction": pred, "reference": ref,
                            "correct": (pred == ref) if ref is not None else None})
            del inputs, output
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard_path.write_text(json.dumps({"rank": rank, "error": traceback.format_exc()}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


def main():
    Path(SHARD_DIR).mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_NAME, cache_dir=HF_CACHE, split="test")
    rows = [dict(x) for x in dataset]
    groups = [list(range(rank, len(rows), GPU_COUNT)) for rank in range(GPU_COUNT)]
    ctx = mp.get_context("spawn")
    processes = [ctx.Process(target=worker, args=(rank, groups[rank], rows)) for rank in range(GPU_COUNT)]
    for p in processes: p.start()
    for p in processes: p.join()
    if any(p.exitcode for p in processes):
        raise RuntimeError("One or more GPU workers failed; inspect sub_logs.")
    merged = []
    for rank in range(GPU_COUNT):
        merged.extend(json.loads((Path(SHARD_DIR) / f"shard_{rank}.json").read_text(encoding="utf-8")))
    merged.sort(key=lambda x: x["index"])
    correct = sum(x.get("correct") is True for x in merged)
    summary = {"model": MODEL_PATH, "dataset": DATASET_NAME, "num_samples": len(merged),
               "correct": correct, "accuracy": correct / len(merged) if merged else 0.0,
               "results": merged}
    Path(RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(RESULT_PATH).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(merged)} results to {RESULT_PATH}; accuracy={summary['accuracy']:.4f}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
