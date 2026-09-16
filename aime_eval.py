"""Eight independent Transformers workers evaluate Qwen3.5-2B on AIME26."""
import json, os, re, traceback
from datetime import datetime
from pathlib import Path
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForImageTextToText, AutoProcessor

MAX_NEW_TOKENS = 32768
ENABLE_THINKING = True
DO_SAMPLE = True
TEMPERATURE = 0.9
TOP_P = 0.95
TOP_K = 20
REPETITION_PENALTY = 1.05
ROLLOUTS_PER_QUESTION = 1
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
        # RTX 2080 Ti is compute capability 7.5 and cannot execute BF16
        # kernels.  Do not rely only on is_bf16_supported(), which can be
        # optimistic with some CUDA/PyTorch combinations.
        capability = torch.cuda.get_device_capability(rank)
        dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16

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
            log(f"model loaded; starting inference (compute_capability={capability}, dtype={dtype}).")
        log(f"assigned {len(assigned_rows)} questions")
        for index, row, rollout in assigned_rows:
            question = str(row.get("problem") or row.get("question"))
            log(f"starting question index={index}, text={question[:100].replace(chr(10), ' ')}")
            try:
                messages = [{"role": "user", "content": question +
                             "\nSolve step by step and put the final integer in \\boxed{...}."}]
                try:
                    prompt = processor.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
                except TypeError:
                    prompt = processor.apply_chat_template(messages, tokenize=False,
                                                            add_generation_prompt=True)
                inputs = processor(text=prompt, return_tensors="pt").to(device)
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                        repetition_penalty=REPETITION_PENALTY,
                        do_sample=DO_SAMPLE,
                        pad_token_id=processor.tokenizer.eos_token_id)
                generated_tokens = output.shape[1] - inputs.input_ids.shape[1]
                response = processor.decode(output[0, inputs.input_ids.shape[1]:],
                                            skip_special_tokens=True)
                pred, gold = extract_answer(response), normalize_gold(row["answer"])
                result = {"index": index, "question": question, "gold": gold,
                          "pred": pred, "correct": pred == gold, "raw": response,
                          "gpu": rank, "rollout": rollout,
                          "generated_tokens": generated_tokens,
                          "hit_token_limit": generated_tokens >= MAX_NEW_TOKENS}
                del inputs, output
                torch.cuda.empty_cache()
            except Exception:
                error_text = traceback.format_exc()
                result = {"index": index, "question": question,
                          "gold": row.get("answer"), "pred": None, "correct": False,
                          "raw": "", "gpu": rank, "rollout": rollout,
                          "generated_tokens": 0, "hit_token_limit": False,
                          "error": error_text}
            result_queue.put(("result", result))
            log(f"finished question index={index}, prediction={result['pred']}, "
                f"gold={result['gold']}, correct={result['correct']}")
            if result["pred"] is None:
                tail = result.get("raw", "")[-400:].replace("\n", " ")
                log(f"answer not found; rollout={rollout}, tokens={result.get('generated_tokens', 0)}, "
                    f"hit_limit={result.get('hit_token_limit', False)}, "
                    f"error={result.get('error', '')[-500:]}, response_tail={tail!r}")
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
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"Detected GPUs={available_gpus}, using GPU_COUNT={GPU_COUNT}", flush=True)
    print(f"Loaded {len(rows)} questions from {DATA_PATH}", flush=True)
    print(f"MAX_NEW_TOKENS={MAX_NEW_TOKENS}", flush=True)
    print(f"ENABLE_THINKING={ENABLE_THINKING}", flush=True)
    print(f"DO_SAMPLE={DO_SAMPLE}", flush=True)
    print(f"TEMPERATURE={TEMPERATURE}", flush=True)
    print(f"TOP_P={TOP_P}", flush=True)
    print(f"TOP_K={TOP_K}", flush=True)
    print(f"REPETITION_PENALTY={REPETITION_PENALTY}", flush=True)
    print(f"ROLLOUTS_PER_QUESTION={ROLLOUTS_PER_QUESTION}", flush=True)
    jobs = [
        (index, row, rollout)
        for index, row in enumerate(rows)
        for rollout in range(ROLLOUTS_PER_QUESTION)
    ]
    assignments = [jobs[rank::GPU_COUNT] for rank in range(GPU_COUNT)]
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
    total_jobs = len(rows) * ROLLOUTS_PER_QUESTION
    while done_workers + len(worker_errors) < GPU_COUNT:
        kind, payload = result_queue.get()
        if kind == "result":
            merged.append(payload)
            correct_so_far = sum(item["correct"] for item in merged)
            print(f"Progress: {len(merged)}/{total_jobs} rollouts, "
                  f"correct={correct_so_far}, latest=GPU{payload['gpu']}/"
                  f"question{payload['index']}/rollout{payload['rollout']}", flush=True)
        elif kind == "done":
            done_workers += 1
        else:
            worker_errors.append(payload)
    for process in processes: process.join()
    failed = [i for i, process in enumerate(processes) if process.exitcode]
    if failed or worker_errors:
        details = "\n".join(item["error"] for item in worker_errors)
        raise RuntimeError(f"GPU workers failed: {failed}\n{details}")
    merged.sort(key=lambda item: (item["index"], item["rollout"]))
    grouped = {index: [] for index in range(len(rows))}
    for item in merged:
        grouped[item["index"]].append(item)
    pass_at_n = sum(any(item["correct"] for item in items) for items in grouped.values())
    majority_items = []
    for index, items in grouped.items():
        votes = {}
        for item in items:
            if item["pred"] is not None:
                votes[item["pred"]] = votes.get(item["pred"], 0) + 1
        majority_pred = max(votes, key=votes.get) if votes else None
        gold = normalize_gold(rows[index]["answer"])
        majority_items.append({"index": index, "gold": gold, "pred": majority_pred,
                               "correct": majority_pred == gold, "votes": votes})
    majority_correct = sum(item["correct"] for item in majority_items)
    report = {"model": MODEL_PATH, "dataset": str(DATA_PATH), "total_questions": len(rows),
              "rollouts_per_question": ROLLOUTS_PER_QUESTION, "gpu_count": GPU_COUNT,
              "pass_at_n_correct": pass_at_n, "pass_at_n_accuracy": pass_at_n / len(rows),
              "majority_correct": majority_correct,
              "majority_accuracy": majority_correct / len(rows),
              "majority_results": majority_items, "rollout_results": merged}
    print(f"AIME26 pass@{ROLLOUTS_PER_QUESTION}: {pass_at_n}/{len(rows)}, "
          f"accuracy={report['pass_at_n_accuracy']:.4f}", flush=True)
    print(f"AIME26 majority@{ROLLOUTS_PER_QUESTION}: {majority_correct}/{len(rows)}, "
          f"accuracy={report['majority_accuracy']:.4f}", flush=True)
    if RESULT_PATH:
        Path(RESULT_PATH).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        print(f"Saved to {RESULT_PATH}", flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    main()
