"""Run Qwen3.5-2B on math-ai/aime26 using vLLM."""

import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams


# Required evaluation parameters: do not change.
max_new_tokens, temperature, enable_thinking = 32768, 0.85, True
top_p, top_k, repetition_penalty = 0.95, 20, 1.05

MODEL_PATH = "/mnt/data/user/zhang_jingdong/models/Qwen3.5-2B"
DATASET_NAME = "math-ai/aime26"
HF_CACHE = "/mnt/data/user/zhang_jingdong/hf_cache"
RESULT_PATH = "/mnt/data/user/zhang_jingdong/NLP_test/eval_result.json"
TENSOR_PARALLEL_SIZE = 8


def prompt(question: str) -> str:
    reasoning = "Think through the problem step by step." if enable_thinking else ""
    return (
        "Solve the following AIME problem. " + reasoning + "\n"
        "Give the final answer as an integer from 0 to 999 in exactly the form "
        r"\boxed{integer}." + "\n\nProblem:\n" + question
    )


def answer_from(text: str):
    """Return the last valid AIME answer found in model output."""
    patterns = [r"\\boxed\s*\{\s*(\d{1,3})\s*\}",
                r"(?:final\s+)?answer\s*[:：]\s*(\d{1,3})\b"]
    for pattern in patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        if found:
            value = int(found[-1])
            if 0 <= value <= 999:
                return value
    return None


def main() -> None:
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")

    data = load_dataset(DATASET_NAME, cache_dir=HF_CACHE)
    split = data["test"] if hasattr(data, "keys") and "test" in data else data
    prompts, gold = [], []
    for row in split:
        question = row.get("problem", row.get("question"))
        if question is None or row.get("answer") is None:
            raise ValueError("Each AIME record must contain problem/question and answer")
        prompts.append(prompt(str(question)))
        gold.append(int(row["answer"]))

    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=temperature,
                              top_p=top_p, top_k=top_k,
                              repetition_penalty=repetition_penalty)
    # CUDA is selected automatically by vLLM; use CUDA_VISIBLE_DEVICES to pin GPUs.
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=TENSOR_PARALLEL_SIZE,
              trust_remote_code=True)
    generated = llm.generate(prompts, sampling)

    results, correct = [], 0
    for index, (row, output, expected) in enumerate(zip(split, generated, gold)):
        raw = output.outputs[0].text if output.outputs else ""
        predicted = answer_from(raw)
        ok = predicted == expected
        correct += int(ok)
        results.append({"index": index, "question": row.get("problem", row.get("question")),
                        "gold": expected, "pred": predicted, "correct": ok, "raw": raw})

    report = {"model": MODEL_PATH, "dataset": DATASET_NAME, "total": len(gold),
              "correct": correct, "accuracy": correct / len(gold), "results": results}
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"AIME26 result: {correct}/{len(gold)} ({report['accuracy']:.4f})")
    print(f"Saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
