"""Eight independent Transformers workers evaluate Qwen3.5-2B on AIME26."""
import json, os, re, traceback
from datetime import datetime
from pathlib import Path
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForImageTextToText, AutoProcessor

max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "8192"))
temperature = float(os.environ.get("TEMPERATURE", "0.8"))
enable_thinking = os.environ.get("ENABLE_THINKING", "1") == "1"
top_p, top_k, repetition_penalty = 0.95, 20, 1.05
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/data/user/zhang_jingdong/models/Qwen3.5-2B")
DATA_PATH = Path(os.environ.get(
    "DATA_PATH", "/mnt/data/user/zhang_jingdong/aime26/aime2026.jsonl"
))
# Respect the server's HF_HOME setting. This avoids unwritable shared-cache
# lock files while retaining a usable default when HF_HOME is unset.
HF_CACHE = os.environ.get("HF_HOME", "/mnt/data/user/zhang_jingdong/hf_cache")
# No files are written unless RESULT_PATH is explicitly supplied.
RESULT_PATH = os.environ.get("RESULT_PATH")
GPU_COUNT = int(os.environ.get("GPU_COUNT", "8"))

def extract_answer(text):
    for pattern in (r"\\boxed\s*\{\s*(\d{1,3})\s*\}",
                    r"\\boxed\s+(\d{1,3})\b",
                    r"(?:final\s+)?answer\s*(?:is|=|[:：])\s*(\d{1,3})\b",
                    r"答案\s*(?:是|为|=|[:：])\s*(\d{1,3})\b"):
        found = re.findall(pattern, text, re.I)
        if found:
            return int(found[-1])
    return None

def normalize_gold(value):
    """Accept integer answers and strings such as ``\\boxed{123}``."""
    if isinstance(value, int):
        return value
    match = re.search(r"\d{1,3}", str(value))
    if not match:
        raise ValueError(f"Cannot parse gold answer: {value!r}")
    return int(match.group())

def load_jsonl(path):
    """Load and minimally validate the local AIME26 JSONL dataset."""
    if not path.is_file():
        raise FileNotFoundError(
            f"AIME26 data file not found: {path}. Set DATA_PATH if it is elsewhere."
        )
    rows = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            question = row.get("problem") or row.get("question")
            if not question:
                raise ValueError(
                    f"Missing 'problem' or 'question' at {path}:{line_number}"
                )
            if "answer" not in row:
                raise ValueError(f"Missing 'answer' at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No questions found in {path}")
    return rows

def worker(rank, assigned_rows, model_load_lock, result_queue):
    def log(message):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] [GPU {rank}] {message}"
        print(line, flush=True)

    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Loading all eight replicas at exactly the same time can create a
        # large transient host/GPU-memory peak. Serialize only model loading;
        # inference starts immediately after this worker releases the lock.
        with model_load_lock:
            torch.cuda.empty_cache()
            log("loading model...")
            processor = AutoProcessor.from_pretrained(
                MODEL_PATH, cache_dir=HF_CACHE, trust_remote_code=True,
                local_files_only=True,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_PATH, cache_dir=HF_CACHE, dtype=dtype,
                trust_remote_code=True, low_cpu_mem_usage=True,
                local_files_only=True,
            ).to(device).eval()
            torch.cuda.empty_cache()
            log("model loaded; starting inference.")
        log(f"assigned {len(assigned_rows)} questions")
        for index, row in assigned_rows:
            question = str(row.get("problem") or row.get("question"))
            log(f"starting question index={index}, text={question[:100].replace(chr(10), ' ')}")
            try:
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
                        repetition_penalty=repetition_penalty,
                        do_sample=temperature > 0,
                        pad_token_id=processor.tokenizer.eos_token_id)
                generated_tokens = output.shape[1] - inputs.input_ids.shape[1]
                response = processor.decode(output[0, inputs.input_ids.shape[1]:],
                                            skip_special_tokens=True)
                pred, gold = extract_answer(response), normalize_gold(row["answer"])
                result = {"index": index, "question": question, "gold": gold,
                          "pred": pred, "correct": pred == gold, "raw": response,
                          "gpu": rank, "generated_tokens": generated_tokens,
                          "hit_token_limit": generated_tokens >= max_new_tokens}
                del inputs, output
                torch.cuda.empty_cache()
            except Exception:
                result = {"index": index, "question": question,
                          "gold": row.get("answer"), "pred": None, "correct": False,
                          "raw": "", "gpu": rank, "error": traceback.format_exc()}
            result_queue.put(("result", result))
            log(f"finished question index={index}, prediction={result['pred']}, "
                f"gold={result['gold']}, correct={result['correct']}")
            if result["pred"] is None and result.get("raw"):
                tail = result["raw"][-400:].replace("\n", " ")
                log(f"answer not found; tokens={result['generated_tokens']}, "
                    f"hit_limit={result['hit_token_limit']}, response_tail={tail!r}")
        result_queue.put(("done", rank))
        log(f"completed all assigned questions: {len(assigned_rows)}")
    except Exception:
        error = traceback.format_exc()
        log(f"worker failed:\n{error}")
        result_queue.put(("error", {"gpu": rank, "error": error}))

def main():
    available_gpus = torch.cuda.device_count()
    if GPU_COUNT < 1 or GPU_COUNT > available_gpus:
        raise RuntimeError(f"GPU_COUNT must be between 1 and {available_gpus}, got {GPU_COUNT}")
    rows = load_jsonl(DATA_PATH)
    print(f"Loaded {len(rows)} questions from {DATA_PATH}", flush=True)
    print(f"Generation: max_new_tokens={max_new_tokens}, "
          f"enable_thinking={enable_thinking}, temperature={temperature}", flush=True)
    indexed_rows = list(enumerate(rows))
    assignments = [indexed_rows[rank::GPU_COUNT] for rank in range(GPU_COUNT)]
    context = mp.get_context("spawn")
    model_load_lock = context.Lock()
    result_queue = context.Queue()
    processes = [context.Process(target=worker,
                                 args=(rank, assignments[rank], model_load_lock,
                                       result_queue),
                                 name=f"aime-gpu-{rank}")
                 for rank in range(GPU_COUNT)]
    for process in processes: process.start()
    merged, worker_errors, done_workers = [], [], 0
    while done_workers + len(worker_errors) < GPU_COUNT:
        kind, payload = result_queue.get()
        if kind == "result":
            merged.append(payload)
            correct_so_far = sum(item["correct"] for item in merged)
            print(f"Progress: {len(merged)}/{len(rows)}, correct={correct_so_far}, "
                  f"latest=GPU{payload['gpu']}/question{payload['index']}", flush=True)
        elif kind == "done":
            done_workers += 1
        else:
            worker_errors.append(payload)
    for process in processes: process.join()
    failed = [i for i, process in enumerate(processes) if process.exitcode]
    if failed or worker_errors:
        details = "\n".join(item["error"] for item in worker_errors)
        raise RuntimeError(f"GPU workers failed: {failed}\n{details}")
    merged.sort(key=lambda item: item["index"])
    correct = sum(item["correct"] for item in merged)
    report = {"model": MODEL_PATH, "dataset": str(DATA_PATH), "total": len(merged),
              "gpu_count": GPU_COUNT, "correct": correct,
              "accuracy": correct / len(merged) if merged else 0.0, "results": merged}
    print(f"AIME26: {correct}/{len(merged)}, accuracy={report['accuracy']:.4f}", flush=True)
    if RESULT_PATH:
        Path(RESULT_PATH).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        print(f"Saved to {RESULT_PATH}", flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    main()
