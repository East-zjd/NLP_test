"""Evaluate Qwen3.5-2B on AIME26 with eight independent single-GPU workers."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


# Required evaluation parameters: do not change.
max_new_tokens, temperature, enable_thinking = 32768, 0.85, True
top_p, top_k, repetition_penalty = 0.95, 20, 1.05

MODEL_PATH = "/mnt/data/user/zhang_jingdong/models/Qwen3.5-2B"
DATASET_NAME = "math-ai/aime26"
HF_CACHE = "/mnt/data/user/zhang_jingdong/hf_cache"
RESULT_PATH = "/mnt/data/user/zhang_jingdong/NLP_test/eval_result.json"
SHARD_DIR = "/mnt/data/user/zhang_jingdong/NLP_test/sub_logs"
GPU_COUNT = 8


def build_prompt(question: str) -> str:
    reasoning = "Think through the problem step by step." if enable_thinking else ""
    return ("Solve the following AIME problem. " + reasoning + "\n"
            "Give the final answer as an integer from 0 to 999 in exactly the form "
            r"\boxed{integer}." + "\n\nProblem:\n" + question)


def extract_answer(text: str):
    for pattern in (r"\\boxed\s*\{\s*(\d{1,3})\s*\}",
                    r"(?:final\s+)?answer\s*[:：]\s*(\d{1,3})\b"):
        values = re.findall(pattern, text, re.IGNORECASE)
        if values and 0 <= int(values[-1]) <= 999:
            return int(values[-1])
    return None


def load_records():
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, cache_dir=HF_CACHE)
    split = dataset["test"] if hasattr(dataset, "keys") and "test" in dataset else dataset
    records = []
    for index, row in enumerate(split):
        question = row.get("problem", row.get("question"))
        answer = row.get("answer")
        if question is None or answer is None:
            raise ValueError("Every AIME record must contain problem/question and answer")
        records.append({"index": index, "question": str(question), "gold": int(answer)})
    return records


def run_worker(worker_id: int) -> None:
    from vllm import LLM, SamplingParams

    records = [record for record in load_records() if record["index"] % GPU_COUNT == worker_id]
    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=temperature,
                              top_p=top_p, top_k=top_k,
                              repetition_penalty=repetition_penalty)
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, trust_remote_code=True,
              max_model_len=max_new_tokens + 4096)
    outputs = llm.generate([build_prompt(record["question"]) for record in records], sampling)
    results = []
    for record, output in zip(records, outputs):
        raw = output.outputs[0].text if output.outputs else ""
        predicted = extract_answer(raw)
        results.append({**record, "pred": predicted,
                        "correct": predicted == record["gold"], "raw": raw})
    Path(SHARD_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(SHARD_DIR) / f"result_gpu_{worker_id}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"GPU {worker_id}: completed {len(results)} problems; saved {path}")


def run_controller() -> None:
    if not Path(MODEL_PATH).is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
    Path(SHARD_DIR).mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).resolve())
    processes = []
    for gpu_id in range(GPU_COUNT):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        command = [sys.executable, script, "--worker", str(gpu_id)]
        log = open(Path(SHARD_DIR) / f"gpu_{gpu_id}.log", "w", encoding="utf-8")
        processes.append((gpu_id, subprocess.Popen(command, env=environment,
                                                    stdout=log, stderr=subprocess.STDOUT), log))
    failed = []
    for gpu_id, process, log in processes:
        code = process.wait()
        log.close()
        if code:
            failed.append(gpu_id)
    if failed:
        raise RuntimeError(f"Workers failed on GPUs {failed}; inspect {SHARD_DIR}/gpu_<id>.log")

    results = []
    for gpu_id in range(GPU_COUNT):
        path = Path(SHARD_DIR) / f"result_gpu_{gpu_id}.json"
        results.extend(json.loads(path.read_text(encoding="utf-8")))
    results.sort(key=lambda item: item["index"])
    correct = sum(item["correct"] for item in results)
    report = {"model": MODEL_PATH, "dataset": DATASET_NAME, "workers": GPU_COUNT,
              "total": len(results), "correct": correct,
              "accuracy": correct / len(results), "results": results}
    Path(RESULT_PATH).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"AIME26 result: {correct}/{len(results)} ({report['accuracy']:.4f})")
    print(f"Saved to: {RESULT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=range(GPU_COUNT))
    args = parser.parse_args()
    run_worker(args.worker) if args.worker is not None else run_controller()


if __name__ == "__main__":
    main()
