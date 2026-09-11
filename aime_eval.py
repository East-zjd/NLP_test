from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
from datasets import load_dataset
import torch

# ===================== 配置区，按需修改 =====================
model_dir = "/mnt/data/user/zhang_jingdong/models/Qwen3.5-2B"
max_new_tokens = 32768
temperature = 0.7
enable_thinking = True
# ==========================================================

# 加载AIME26数据集
print("Loading AIME26 dataset from HuggingFace...")
dataset = load_dataset("math-ai/aime26", cache_dir="/mnt/data/user/zhang_jingdong/hf_cache")
aime_data = dataset["test"]
print(f"AIME26 total test problems: {len(aime_data)}")

# 加载模型与分词器
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True
)
print("✅ Model loaded!")

# 构造prompt，要求模型先推理，最后输出【Answer:数字】
def build_prompt(question: str):
    if enable_thinking:
        prompt = f"""Please solve the following math problem step by step.
Put your final answer strictly in the format:【Answer:XXX】. The answer is an integer between 0 and 999.
Problem:
{question}
"""
    else:
        prompt = f"""Solve this math problem, output final answer strictly as 【Answer:XXX】.
Problem:
{question}
"""
    return prompt

# 从文本提取答案数字
def extract_answer(text: str):
    match = re.search(r"【Answer:(\d+)】", text)
    if match:
        return int(match.group(1))
    return None

# 评测循环
correct = 0
total = len(aime_data)
result_log = []

for idx, item in enumerate(aime_data):
    q = item["problem"]
    gold_ans = int(item["answer"])
    prompt = build_prompt(q)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"\n===== Question {idx+1}/{total} =====")
    print(f"Q: {q}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred_ans = extract_answer(raw_output)

    print(f"Predicted answer: {pred_ans}, Gold answer: {gold_ans}")

    if pred_ans == gold_ans:
        correct +=1
        print("✅ Correct")
    else:
        print("❌ Wrong")
    result_log.append({
        "question": q,
        "gold": gold_ans,
        "pred": pred_ans,
        "raw": raw_output
    })

# 汇总结果
acc = correct / total
print("\n" + "="*50)
print(f"Total:{total}, Correct:{correct}, Accuracy = {acc:.4f}")

# 保存日志到你Ceph目录，不会丢失
with open("/mnt/data/user/zhang_jingdong/NLP_test/eval_result.json","w",encoding="utf-8") as f:
    json.dump(result_log, f, ensure_ascii=False, indent=2)
print("📝 Result saved to eval_result.json")
