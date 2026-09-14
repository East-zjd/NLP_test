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
temperature = 0.85
enable_thinking = True
top_p = 0.95
top_k = 20
repetition_penalty = 1.05
# 最终合并结果路径
final_result_path = "/mnt/data/user/zhang_jingdong/NLP_test/eval_result.json"
# 子进程分片日志保存目录
sub_log_dir = "/mnt/data/user/zhang_jingdong/NLP_test/sub_logs"
# ==========================================================
# 构造prompt，要求模型输出boxed答案（AIME标准格式）
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

# 多格式答案提取，解决pred=None
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


def worker(gpu_id, sub_data, queue: Queue):
    """子进程任务：接收主进程传好的分片数据，不再联网加载数据集"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU{gpu_id}] Start worker, handle {len(sub_data)} problems")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print(f"[GPU{gpu_id}] Model loaded successfully.")
    correct = 0
    total_sub = len(sub_data)
    sub_log = []

    for item in sub_data:
        q = item["problem"]
        gold_ans = int(item["answer"])
        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print(f"\n===== GPU{gpu_id} | Question =====")
        print(f"Q: {q[:120]}...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
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

    os.makedirs(sub_log_dir, exist_ok=True)
    sub_save_path = os.path.join(sub_log_dir, f"gpu{gpu_id}_subresult.json")
    with open(sub_save_path, "w", encoding="utf-8") as f:
        json.dump(sub_log, f, ensure_ascii=False, indent=2)
    print(f"[GPU{gpu_id}] Sub log saved to {sub_save_path}, correct={correct}/{total_sub}")
    queue.put({
        "gpu": gpu_id,
        "correct": correct,
        "total": total_sub,
        "log_path": sub_save_path
    })


if __name__ == "__main__":
    from multiprocessing import Process, Queue
    print("Loading AIME26 dataset ONCE in main process...")
    # ==========主进程一次性加载数据集，只访问一次hf！==========
    dataset = load_dataset("math-ai/aime26", cache_dir="/mnt/data/user/zhang_jingdong/hf_cache")
    aime_data = dataset["test"]
    total_problems = len(aime_data)
    print(f"AIME26 total test problems: {total_problems}")

    num_gpus = 8
    # 切分数据集分片
    chunks = []
    chunk_size = total_problems // num_gpus
    remain = total_problems % num_gpus
    s = 0
    for g in range(num_gpus):
        add = 1 if g < remain else 0
        e = s + chunk_size + add
        sub_data = aime_data.select(range(s,e))
        chunks.append((g, sub_data))
        s = e
    print(f"Task split finished.")

    q = Queue()
    proc_list = []
    for gpu_id, sub_data in chunks:
        p = Process(target=worker, args=(gpu_id, sub_data, q))
        p.start()
        proc_list.append(p)

    finished = 0
    total_correct = 0
    total_cnt = 0
    all_log = []
    while finished < num_gpus:
        res = q.get()
        finished += 1
        total_correct += res["correct"]
        total_cnt += res["total"]
        with open(res["log_path"], "r", encoding="utf-8") as f:
            part_data = json.load(f)
            all_log.extend(part_data)
    # 等待所有进程结束
    for p in proc_list:
        p.join()

    acc = total_correct / total_cnt
    print("\n" + "="*60)
    print(f"Total:{total_cnt}, Correct:{total_correct}, Accuracy = {acc:.4f}")
    # 保存合并结果
    with open(final_result_path, "w", encoding="utf-8") as f:
        json.dump(all_log, f, ensure_ascii=False, indent=2)
    print(f"📝 Merged full result saved to {final_result_path}")
    # 统计None数量，方便调试
    none_count = sum(1 for item in all_log if item["pred"] is None)
    print(f"📌 Number of None predictions: {none_count}/{total_cnt}")
