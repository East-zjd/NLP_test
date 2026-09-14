"""Eight independent Transformers workers evaluate Qwen3.5-2B on AIME26."""
import json, re, traceback
from pathlib import Path
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

max_new_tokens, temperature, enable_thinking = 32768, 0.85, True
top_p, top_k, repetition_penalty = 0.95, 20, 1.05
MODEL_PATH = "/mnt/data/user/zhang_jingdong/models/Qwen3.5-2B"
HF_CACHE = "/mnt/data/user/zhang_jingdong/hf_cache"
RESULT_PATH = "/mnt/data/user/zhang_jingdong/NLP_test/eval_result.json"
SHARD_DIR = Path("/mnt/data/user/zhang_jingdong/NLP_test/sub_logs")
GPU_COUNT = 8

def extract_answer(text):
    for pattern in (r"\\boxed\s*\{\s*(\d{1,3})\s*\}",
                    r"(?:final\s+)?answer\s*[:：]\s*(\d{1,3})\b"):
        found = re.findall(pattern, text, re.I)
        if found:
            return int(found[-1])
    return None

def worker(rank, rows, model_load_lock):
    path = SHARD_DIR / f"shard_{rank}.json"
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # Loading all eight replicas at exactly the same time can create a
        # large transient host/GPU-memory peak. Serialize only model loading;
        # inference starts immediately after this worker releases the lock.
        with model_load_lock:
            torch.cuda.empty_cache()
            print(f"GPU {rank}: loading model...", flush=True)
            processor = AutoProcessor.from_pretrained(
                MODEL_PATH, cache_dir=HF_CACHE, trust_remote_code=True,
                local_files_only=True,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_PATH, cache_dir=HF_CACHE, dtype=torch.bfloat16,
                trust_remote_code=True, low_cpu_mem_usage=True,
                local_files_only=True,
            ).to(device).eval()
            torch.cuda.empty_cache()
            print(f"GPU {rank}: model loaded; starting inference.", flush=True)
        results = []
        for index in range(rank, len(rows), GPU_COUNT):
            row = rows[index]
            question = str(row.get("problem") or row.get("question"))
            messages = [{"role": "user", "content": question +
                         "\nSolve step by step and put the final integer in \\boxed{...}."}]
            try:
                prompt = processor.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=enable_thinking)
            except TypeError:
                prompt = processor.apply_chat_template(messages, tokenize=False,
                                                        add_generation_prompt=True)
            inputs = processor(text=prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    repetition_penalty=repetition_penalty, do_sample=True,
                    pad_token_id=processor.tokenizer.eos_token_id)
            response = processor.decode(output[0, inputs.input_ids.shape[1]:],
                                        skip_special_tokens=True)
            pred, gold = extract_answer(response), int(row["answer"])
            results.append({"index": index, "question": question, "gold": gold,
                            "pred": pred, "correct": pred == gold, "raw": response})
            del inputs, output
            torch.cuda.empty_cache()
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        path.write_text(json.dumps({"gpu": rank, "error": traceback.format_exc()},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        raise

def main():
    if torch.cuda.device_count() < GPU_COUNT:
        raise RuntimeError(f"Need {GPU_COUNT} GPUs, found {torch.cuda.device_count()}")
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("math-ai/aime26", cache_dir=HF_CACHE)
    split = dataset["test"] if "test" in dataset else dataset[next(iter(dataset.keys()))]
    rows = [dict(row) for row in split]
    context = mp.get_context("spawn")
    model_load_lock = context.Lock()
    processes = [context.Process(target=worker, args=(rank, rows, model_load_lock))
                 for rank in range(GPU_COUNT)]
    for process in processes: process.start()
    for process in processes: process.join()
    failed = [i for i, process in enumerate(processes) if process.exitcode]
    if failed:
        raise RuntimeError(f"GPU workers failed: {failed}; inspect {SHARD_DIR}/shard_<id>.json")
    merged = []
    for rank in range(GPU_COUNT):
        merged.extend(json.loads((SHARD_DIR / f"shard_{rank}.json").read_text(encoding="utf-8")))
    merged.sort(key=lambda item: item["index"])
    correct = sum(item["correct"] for item in merged)
    report = {"model": MODEL_PATH, "dataset": "math-ai/aime26", "total": len(merged),
              "correct": correct, "accuracy": correct / len(merged), "results": merged}
    Path(RESULT_PATH).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"AIME26: {correct}/{len(merged)}, accuracy={report['accuracy']:.4f}")
    print(f"Saved to {RESULT_PATH}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
