from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
from datasets import load_dataset
import torch
import os
from multiprocessing import Queue


# ===================== 配置区，按需修改 =====================
model_dir = "/mnt/data/user/zhang_jingdong/models/Qwen3.5-2B"
max_new_tokens = 32768
temperature = 0.7
enable_thinking = True
# 最终合并结果路径
final_result_path = "/mnt/data/user/zhang_jingdong/NLP_test/eval_result.json"
# 子进程分片日志保存目录
sub_log_dir = "/mnt/data/user/zhang_jingdong/NLP_test/sub_logs"
# ==========================================================

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


def worker(gpu_id, start_idx, end_idx, queue: Queue):
    """子进程任务：单卡，处理[start_idx, end_idx)题目，本地保存分片日志"""
    # 绑定当前进程到指定GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU{gpu_id}] Start worker, handle problem range: [{start_idx}, {end_idx})")
    # 加载数据集
    dataset = load_dataset("math-ai/aime26", cache_dir="/mnt/data/user/zhang_jingdong/hf_cache")
    aime_data = dataset["test"]

    # 子进程内部加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print(f"[GPU{gpu_id}] Model loaded successfully.")

    correct = 0
    total_sub = end_idx - start_idx
    sub_log = []

    for idx in range(start_idx, end_idx):
        item = aime_data[idx]
        q = item["problem"]
        gold_ans = int(item["answer"])
        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print(f"\n===== GPU{gpu_id} | Question {idx+1} =====")
        print(f"Q: {q[:120]}...") # 只打印题目开头，防止刷屏

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_ans = extract_answer(raw_output)

        print(f"[GPU{gpu_id}] Predicted answer: {pred_ans}, Gold answer: {gold_ans}")
        if pred_ans == gold_ans:
            correct += 1
            print("✅ Correct")
        else:
            print("❌ Wrong")

        sub_log.append({
            "question": q,
            "gold": gold_ans,
            "pred": pred_ans,
            "raw": raw_output
        })
        # 每做完一题也可以写盘，防止崩溃丢数据（可选）
        # with open(os.path.join(sub_log_dir,f"gpu{gpu_id}_tmp.json"),"w",encoding="utf-8") as f:
        #     json.dump(sub_log,f,ensure_ascii=False,indent=2)

    # 子进程跑完，保存分片日志
    os.makedirs(sub_log_dir, exist_ok=True)
    sub_save_path = os.path.join(sub_log_dir, f"gpu{gpu_id}_subresult.json")
    with open(sub_save_path, "w", encoding="utf-8") as f:
        json.dump(sub_log, f, ensure_ascii=False, indent=2)
    print(f"[GPU{gpu_id}] Sub log saved to {sub_save_path}, correct={correct}/{total_sub}")

    # 将统计信息送回主进程用于汇总准确率
    queue.put({
        "gpu": gpu_id,
        "correct": correct,
        "total": total_sub,
        "log_path": sub_save_path
    })


if __name__ == "__main__":
    from multiprocessing import Process, Queue
    total_problems = 30
    num_gpus = 8

    # 切分30题到8张卡
    chunks = []
    chunk_size = total_problems // num_gpus
    remain = total_problems % num_gpus
    s = 0
    for g in range(num_gpus):
        add = 1 if g < remain else 0
        e = s + chunk_size + add
        chunks.append((g, s, e))
        s = e

    print(f"Task split: {chunks}")
    q = Queue()
    proc_list = []

    # 启动8个子进程
    for gpu_id, start, end in chunks:
        p = Process(target=worker, args=(gpu_id, start, end, q))
        p.start()
        proc_list.append(p)

    # 收集子进程统计信息
    finished = 0
    total_correct = 0
    total_cnt = 0
    all_log = []

    while finished < num_gpus:
        res = q.get()
        finished += 1
        total_correct += res["correct"]
        total_cnt += res["total"]
        # 读取分片日志，合并
        with open(res["log_path"], "r", encoding="utf-8") as f:
            part_data = json.load(f)
        all_log.extend(part_data)

    # 等待全部进程退出
    for p in proc_list:
        p.join()

    acc = total_correct / total_cnt
    print("\n" + "="*60)
    print(f"Total:{total_cnt}, Correct:{total_correct}, Accuracy = {acc:.4f}")
    # 保存合并后的完整结果
    with open(final_result_path, "w", encoding="utf-8") as f:
        json.dump(all_log, f, ensure_ascii=False, indent=2)
    print(f"📝 Merged full result saved to {final_result_path}")
