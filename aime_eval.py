"""Evaluate Qwen3.5-2B on AIME26 with vLLM."""
import json, os, re
from typing import Any, Dict, Optional
from datasets import load_dataset
from vllm import LLM, SamplingParams

model_dir = "/mnt/data/user/zhang_jingdong/models/Qwen3.5-2B"
max_new_tokens, temperature, enable_thinking = 32768, 0.85, True
top_p, top_k, repetition_penalty = 0.95, 20, 1.05
final_result_path = "/mnt/data/user/zhang_jingdong/NLP_test/eval_result.json"
sub_log_dir = "/mnt/data/user/zhang_jingdong/NLP_test/sub_logs"
tensor_parallel_size, load_in_8bit = 8, True

def build_prompt(question: str) -> str:
    thinking = "Show your reasoning step by step before the answer." if enable_thinking else ""
    return ("You are solving an AIME mathematics problem.\n" + thinking + "\n"
            "Return the final answer as an integer from 0 to 999, enclosed exactly in "
            "\\boxed{...}.\n\nProblem:\n" + question + "\n")

def extract_answer(text: str) -> Optional[int]:
    for pattern in (r"\\boxed\s*\{\s*(\d{1,3})\s*\}",
                    r"(?:final\s+answer|answer)\s*[:：]\s*\**\s*(\d{1,3})\b",
                    r"\b(\d{1,3})\s*(?:is\s+the\s+)?final\s+answer\b"):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches and 0 <= int(matches[-1]) <= 999:
            return int(matches[-1])
    return None

def main() -> None:
    dataset = load_dataset("math-ai/aime26", cache_dir="/mnt/data/user/zhang_jingdong/hf_cache")
    split = dataset["test"] if hasattr(dataset, "keys") and "test" in dataset else dataset
    prompts, records = [], []
    for item in split:
        question = item.get("problem") or item.get("question")
        if question is None or item.get("answer") is None:
            raise KeyError("AIME item must contain problem/question and answer")
        prompts.append(build_prompt(str(question)))
        records.append({"question": str(question), "gold": int(item["answer"])})
    params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                            top_k=top_k, repetition_penalty=repetition_penalty)
    llm = LLM(model=model_dir, tensor_parallel_size=tensor_parallel_size,
              load_in_8bit=load_in_8bit, trust_remote_code=True, device="cuda")
    results, correct = [], 0
    for record, output in zip(records, llm.generate(prompts, params)):
        raw = output.outputs[0].text if output.outputs else ""
        pred = extract_answer(raw); ok = pred == record["gold"]; correct += int(ok)
        results.append({**record, "pred": pred, "correct": ok, "raw": raw})
    total = len(results); accuracy = correct / total
    print(f"AIME26: {correct}/{total} correct, accuracy={accuracy:.4f}")
    os.makedirs(os.path.dirname(final_result_path) or ".", exist_ok=True); os.makedirs(sub_log_dir, exist_ok=True)
    with open(final_result_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_dir, "dataset": "math-ai/aime26", "total": total,
                   "correct": correct, "accuracy": accuracy, "results": results}, f,
                  ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
