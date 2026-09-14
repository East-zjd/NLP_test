import re
import json
import os
from datasets import load_dataset
from vllm import LLM, SamplingParams

# ===================== 配置区，完全沿用你原来的参数 =====================
model_dir = "/mnt/data/user/zhang_jingdong/models/Qwen3.5-2B"
max_new_tokens = 32768
temperature = 0.85
enable_thinking = True
top_p = 0.95
top_k = 20
repetition_penalty = 1.05
# 最终合并结果路径
final_result_path = "/mnt/data/user/zhang_jingdong/NLP_test/eval_result.json"
sub_log_dir = "/mnt/data/user/zhang_jingdong/NLP_test/sub_logs"
tensor_parallel_size = 8  # 张量并行，切分到全部8张卡
load_in_8bit = True
# ======================================================================

# 构造prompt，和原版完全一致
def build_prompt(question: str):
    if enable_thinking:
        prompt = """Please solve the following math problem step by step.
Put your final answer strictly inside \\boxed{integer}. The answer is an integer between 0 and 999.
Problem:
%s
""" % question
    else:
        prompt = """Solve this math problem, put final answer strictly inside \\boxed{integer}.
Problem:
%s
""" % question
    return prompt

# 答案提取函数，完全复用原版逻辑
def extract_answer(text: str):
    # 优先匹配AIME标准boxed
    boxed_match = re.search(r"\\boxed\{(\d+)\}", text)
    if boxed_match:
        return int(boxed_match.group(1))
    # 匹配【Answer:XXX】
    bracket_match = re.search(r"【Answer:(\d+)】", text)
    if bracket_match:
        return int(bracket_match.group(1))
    # 匹配英文Answer:数字
    ans_match = re.search(r"Answer[:：]\s*(\d+)", text)
    if ans_match:
        return int(ans_match.group(1))
    return None


if __name__ == "__main__":
    print("Loading AIME26 dataset...")
    dataset = load_dataset("math-ai/aime26", cache_dir="/mnt/data/user/zhang_jingdong/hf_cache")
    aime_data = dataset["test"]
    total_problems = len(aime_data)
    print(f"AIME26 total test problems: {total_problems}")

    # vLLM采样参数，对齐transformers generate参数
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    # 初始化vLLM引擎：张量并行8卡 + 8bit量化
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tensor_parallel_size,
        load_in_8bit=load_in_8bit,
        trust_remote_code=True,
        device="cuda",
    )

    # 批量构造prompt列表
    prompts = []
    gold_answers = []
    raw_items = []
    for item in aime_data:
        q = item["problem"]
        gold_ans = int(item["answer"])
        prompt = build_prompt(q)
        prompts.append(prompt)
        gold_answers.append(gold_ans)
        raw_items.append(item)

    print("Start batch inference with tensor parallel 8 GPUs...")
    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    all_log = []
    for idx, output in enumerate(outputs):
        q = raw_items[idx]["problem"]
        gold_ans = gold_answers[idx]
        raw_output = output.outputs[0].text
        pred_ans = extract_answer(raw_output)
        print(f"\n===== Question {idx} =====")
        print(f"Q: {q[:120]}...")
        print(f"Predicted answer: {pred_ans}, Gold answer: {gold_ans}")
        if pred_ans == gold_ans:
            correct += 1
            print("✅ Correct")
        else:
            print("❌ Wrong")
        all_log.append({
            "question": q,
            "gold": gold_ans,
            "pred": pred_ans,
            "raw": raw_output
        })

    acc = correct / total_problems
    print("\n" + "="*60)
    print(f"Total:{total_problems}, Correct:{correct}, Accuracy = {acc:.4f}")
    os.makedirs(sub_log_dir, exist_ok=True)
    with open(final_result_path, "w", encoding="utf-8") as f:
        json.dump(all_log, f, ensure_ascii=False, indent=2)
    print(f"📝 Merged full result saved to {final_result_path}")
    none_count = sum(1 for item in all_log if item["pred"] is None)
    print(f"📌 Number of None predictions: {none_count}/{total_problems}")
