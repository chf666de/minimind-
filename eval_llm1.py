import time
import argparse
import warnings
import torch
import re
import json
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


def load_questions(question_file_path):
    try:
        with open(question_file_path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f.readlines() if line.strip()]
        return questions
    except FileNotFoundError:
        print(f"[错误] 问题文件 {question_file_path} 未找到。")
        exit(1)


def load_scoring_rules(file_path):

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f'[错误] 加载评分规则文件失败：{e}')
        return ""


def extract_score_from_text(text):

    patterns = [
        r'(\d{1,3}(\.\d+)?)\s*分',
        r'分数[：:]\s*(\d{1,3}(\.\d+)?)',
        r'最终评分为[：:]\s*(\d{1,3}(\.\d+)?)',
        r'\b(\d{1,3}(\.\d+)?)\s*/\s*100\b',
        r'\b(\d{1,3}(\.\d+)?)\b(?![.]*\d)'
    ]
    for pattern in patterns:
        matches = re.search(pattern, text)
        if matches:
            try:
                score = float(matches.group(1))
                if 0 <= score <= 100:
                    return score
            except ValueError:
                continue
    return None


def get_score_via_api(prompt, response, scoring_rules, max_retries=3):

    scoring_input = f"请根据以下评分规则对模型的回答进行评分（满分100分）：\n\n"
    scoring_input += f"【评分规则】\n{scoring_rules}\n\n"
    scoring_input += f"【需要评分的回答】\n问题：{prompt}\n回答：{response}\n\n"
    scoring_input += "请首先给出具体的评分分数（一个0到100之间的数字），然后提供简要的评分理由。"

    messages = [
        {"role": "system",
         "content": "你是一个专业的文本质量评估助手，需要严格根据用户提供的评分规则进行客观、准确的评分。你的输出必须包含一个明确的分数。"},
        {"role": "user", "content": scoring_input}
    ]

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="qwen2.5-72b-instruct",  # 可在此处更换评测模型
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                stream=False
            )
            scoring_output_text = completion.choices[0].message.content
            extracted_score = extract_score_from_text(scoring_output_text)
            return scoring_output_text, extracted_score
        except Exception as e:
            print(f'[API评分第{attempt + 1}次尝试失败] 错误: {e}')
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print('[API评分失败] 已达到最大重试次数。')
                return f"API调用失败：{str(e)}", None


def generate_answers(prompts, model, tokenizer, args):

    print("[阶段一] 正在使用本地模型生成答案...")
    conversation = []
    all_generated_answers = []
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if args.show_speed else None

    for idx, prompt in enumerate(tqdm(prompts, desc="生成答案")):

        setup_seed(2026)  # 或使用随机种子 setup_seed(random.randint(0, 2048))

        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason':
            templates["enable_thinking"] = True  
        try:
            model_inputs = tokenizer.apply_chat_template(**templates) if hasattr(tokenizer,
                                                                                 'apply_chat_template') else (
                        tokenizer.bos_token + prompt)
        except Exception:
            # 回退方案
            model_inputs = prompt

        inputs = tokenizer(model_inputs, return_tensors="pt", truncation=True).to(args.device)

        print(f'\n💬 问题 {idx + 1}: {prompt}') if not args.quiet else None
        print('🤖 回答: ', end='') if not args.quiet and streamer else None

        st = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True if args.temperature > 0 else False,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=1.0
            )
        gen_time = time.time() - st

        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})


        answer_record = {
            "question_id": idx + 1,
            "question": prompt,
            "answer": response,
            "generation_time": gen_time
        }
        all_generated_answers.append(answer_record)

        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        if args.show_speed and not args.quiet:
            print(f'\n[速度]: {gen_tokens / gen_time:.2f} tokens/s\n') if gen_time > 0 else print('\n')
        elif not args.quiet:
            print(f'{response}\n') 
    return all_generated_answers


def plot_score_line_chart(scores, output_dir, timestamp):
   
    if not scores:
        print("[警告] 没有有效的分数可绘制图表。")
        return

    plt.figure(figsize=(12, 6))
    indices = list(range(1, len(scores) + 1))
    plt.plot(indices, scores, marker='o', linestyle='-', linewidth=2, markersize=8)

    plt.title('API模型评分结果折线图', fontsize=15, pad=15)
    plt.xlabel('问题序号', fontsize=12)
    plt.ylabel('评分分数', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(indices)
    plt.ylim(0, 105)  # 设置y轴范围，为满分100留出空间

    # 在每个数据点上标注分数
    for i, score in enumerate(scores):
        plt.text(indices[i], score + 1, f'{score:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, f'score_line_chart_{timestamp}.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[图表保存] 评分折线图已保存至: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="本地模型生成 + API模型评分一体化评测脚本")

    # === 模型加载参数 (来自文档1和文档2) ===
    parser.add_argument('--load_from', default='model', type=str, help="主模型加载路径（'model'或具体路径）")
    parser.add_argument('--save_dir', default='out', type=str, help="（若用自定义权重）权重保存目录")
    parser.add_argument('--weight', default='pretrain', type=str, help="（若用自定义权重）权重名称前缀")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=1, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")

    # === 生成参数 (来自文档2) ===
    parser.add_argument('--max_new_tokens', default=2048, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--quiet', default=0, type=int, help="安静模式（1=不打印生成过程，仅保存）")

    # === 评测与文件参数 (来自文档1并调整) ===
    parser.add_argument('--question_file', type=str, default='D:\\python\\minimind-master\\eval_outputs\\generate_30.txt',required=False, help='评测问题文本文件路径（每行一个问题）')
    parser.add_argument('--scoring_rules_file', type=str, default='D:\\python\\minimind-master\\eval_outputs\\scoring_rules.txt',required=False, help='评分规则文本文件路径')
    parser.add_argument('--output_dir', type=str, default='./eval_outputs/eval_data', help='所有输出文件的保存目录')
    parser.add_argument('--api_model', type=str, default='qwen3.5-plus', help='API评分模型名称')

    args = parser.parse_args()

    # 0. 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # 1. 初始化本地模型
    print(f"[初始化本地模型中...] 加载自: {args.load_from}")
    model, tokenizer = init_model(args)

    # 2. 加载评测问题与评分规则
    prompts = load_questions(args.question_file)
    scoring_rules = load_scoring_rules(args.scoring_rules_file)
    print(f"[加载完成] 共加载 {len(prompts)} 个问题。")
    print(f"[加载完成] 评分规则字符数: {len(scoring_rules)}")

    # 3. 阶段一：生成答案
    generated_answers = generate_answers(prompts, model, tokenizer, args)

    # 保存生成的答案到文本文件 (任务要求)
    answers_text_path = os.path.join(args.output_dir, f'generated_answers_{timestamp}.txt')
    with open(answers_text_path, 'w', encoding='utf-8') as f:
        for item in generated_answers:
            f.write(f"问题 {item['question_id']}: {item['question']}\n")
            f.write(f"回答: {item['answer']}\n")
            f.write("-" * 50 + "\n")
    print(f"[阶段一完成] 所有生成的答案已保存至: {answers_text_path}")

    # 保存生成的答案到JSON文件 (便于程序读取)
    answers_json_path = os.path.join(args.output_dir, f'generated_answers_{timestamp}.json')
    with open(answers_json_path, 'w', encoding='utf-8') as f:
        json.dump(generated_answers, f, ensure_ascii=False, indent=2)
    print(f"[结果保存] 生成答案(JSON格式)已保存至: {answers_json_path}")

    # 4. 阶段二：API评分
    print("\n[阶段二] 正在使用API模型进行评分...")
    all_scoring_outputs = []
    all_scores = []

    for item in tqdm(generated_answers, desc="API评分"):
        prompt = item['question']
        response = item['answer']

        scoring_st = time.time()
        scoring_output, extracted_score = get_score_via_api(prompt, response, scoring_rules)
        scoring_time = time.time() - scoring_st

        scoring_record = {
            "question_id": item['question_id'],
            "question": prompt,
            "answer": response,
            "scoring_output": scoring_output,
            "extracted_score": extracted_score,
            "scoring_time": scoring_time
        }
        all_scoring_outputs.append(scoring_record)


    scoring_text_path =os.path.join(args.output_dir, f'api_scoring_outputs_{timestamp}.txt')
    with open(scoring_text_path, 'w', encoding='utf-8') as f:
        for record in all_scoring_outputs:
            f.write(f"问题 {record['question_id']}: {record['question']}\n")
            f.write(f"评分输出:\n{record['scoring_output']}\n")
            f.write(f"提取分数: {record['extracted_score']}\n")
            f.write("=" * 60 + "\n")
    print(f"[阶段二完成] API评分输出文本已保存至: {scoring_text_path}")

    # 保存评分输出到JSON
    scoring_json_path = os.path.join(args.output_dir, f'api_scoring_outputs_{timestamp}.json')
    with open(scoring_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_scoring_outputs, f, ensure_ascii=False, indent=2)
    print(f"[结果保存] 评分输出(JSON格式)已保存至: {scoring_json_path}")

    # 计算并保存所有问题的平均分
    print("\n[新增功能] 正在计算并保存平均分...")

    # 1. 从all_scoring_outputs中提取extracted_score和question_id
    score_dict = {}
    total_score = 0
    valid_scores_count = 0

    for record in all_scoring_outputs:
        question_id = record['question_id']
        extracted_score = record['extracted_score']

        if extracted_score is not None:
            score_dict[question_id] = extracted_score
            total_score += extracted_score
            valid_scores_count += 1

    # 2. 计算平均分
    if valid_scores_count > 0:
        all_score = total_score / valid_scores_count
        max_question_id = max(score_dict.keys()) if score_dict else 0

        print(f"  有效评分数量: {valid_scores_count}/{len(all_scoring_outputs)}")
        print(f"  最大问题ID: {max_question_id}")
        print(f"  总分: {total_score:.2f}")
        print(f"  平均分(all_score): {all_score:.2f}")

        # 3. 从ckp提取模型名称
        # ckp变量在init_model函数中定义，需要获取它
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model_name = os.path.basename(ckp).replace('.pth', '')

        # 如果有LoRA权重，添加到模型名称中
        if args.lora_weight != 'None':
            model_name = f"{model_name}_lora_{args.lora_weight}"

        print(f"  提取的模型名称: {model_name}")

        # 4. 创建专门的汇总文件夹并保存结果
        summary_dir = "./eval_outputs/score_summary"
        os.makedirs(summary_dir, exist_ok=True)

        summary_file = os.path.join(summary_dir, "model_scores_summary.csv")

        # 检查文件是否存在，如果不存在则创建并写入表头
        file_exists = os.path.isfile(summary_file)

        with open(summary_file, 'a', encoding='utf-8') as f:
            if not file_exists:
                # 写入CSV表头
                f.write(
                    "timestamp,model_name,all_score,questions_count,valid_scores_count,total_score,max_question_id\n")

            # 写入数据行
            f.write(
                f"{timestamp},{model_name},{all_score:.4f},{len(all_scoring_outputs)},{valid_scores_count},{total_score:.2f},{max_question_id}\n")

        print(f"[平均分保存] 模型 '{model_name}' 的平均分 {all_score:.2f} 已保存至: {summary_file}")

        # 可选：同时保存详细分数字典到JSON文件
        detailed_scores_file = os.path.join(summary_dir, f"{model_name}_{timestamp}_detailed_scores.json")
        detailed_data = {
            "model_name": model_name,
            "timestamp": timestamp,
            "all_score": all_score,
            "questions_count": len(all_scoring_outputs),
            "valid_scores_count": valid_scores_count,
            "total_score": total_score,
            "max_question_id": max_question_id,
            "score_dict": score_dict,
            "all_scoring_outputs": all_scoring_outputs  # 包含所有原始数据
        }

        with open(detailed_scores_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)

        print(f"[详细数据保存] 详细评分数据已保存至: {detailed_scores_file}")

        # 5. 绘制并保存平均分图表
        plot_score_line_chart([score_dict.get(i) for i in sorted(score_dict.keys())], args.output_dir, timestamp)

    else:
        print("  警告：没有有效的评分数据可用于计算平均分。")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    main()
