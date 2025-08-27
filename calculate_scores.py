import os
import json
import pandas as pd
from bert_score import score
import logging
from datetime import datetime
import re
from concurrent.futures import ProcessPoolExecutor
import torch
import multiprocessing

log_file = f"score_calculation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

task_mapping = {
    "法条引用": "Legal Provision Citation",
    "解决方案": "Solution Generation",
    "问题类别": "Issue Classification",
    "争议焦点": "Dispute Focus Identification",
    "整体回复": "Comprehensive Response"
}

GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0


def init_worker(worker_id, gpu_count):
    if gpu_count > 0:
        gpu_id = worker_id % gpu_count
        torch.cuda.set_device(gpu_id)
        logging.info(f"进程 {worker_id} 分配到 GPU: cuda:{gpu_id}")


def get_models_and_tasks(results_dir="./results"):
    try:
        if not os.path.exists(results_dir):
            logging.error(f"结果目录 {results_dir} 不存在")
            return [], []
        models = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d != "logs"]
        tasks = list(task_mapping.keys())
        logging.info(f"发现 {len(models)} 个模型: {', '.join(models)}")
        logging.info(f"任务: {', '.join(tasks)}")
        return models, tasks
    except Exception as e:
        logging.error(f"获取模型和任务名称失败: {str(e)}")
        return [], []


def extract_option(text):
    if not text or not isinstance(text, str):
        return None
    patterns = [
        r'(?:答案是|选项(?:是)?|最合适的选项是|符合的选项是|问题属于|根据.*?(?:选项是|归类为|类别是))?\s*(?:["\'\[]|\(\s*(?:Option|选项)\s*)?\s*([A-Da-d])\s*(?:["\'\]])?(?:\.|\s|:|\)|$|[^A-Da-d])',
        r'\b([A-Da-d])(?:\.|\s|:|$)',
        r'["\']([A-Da-d])["\'](?:\.|\s|:|$)',
        r'\[\s*([A-Da-d])\s*\]',
        r'\(\s*(?:Option|选项)\s*([A-Da-d])\s*\)',
    ]
    matches = set()
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            matches.add(match.group(1).upper())

    if len(matches) == 1:
        return list(matches)[0]
    elif len(matches) > 1:
        return None
    return None


def calculate_accuracy(true_answers, pred_answers):
    if not true_answers or not pred_answers:
        logging.warning("真实答案或预测答案为空，返回 0.0")
        return 0.0
    try:
        correct = 0
        for i, (true, pred) in enumerate(zip(true_answers, pred_answers)):
            extracted_option = extract_option(pred)
            if extracted_option and extracted_option == str(true).strip().upper():
                correct += 1
            else:
                logging.debug(f"样本 {i}: 提取失败: predicted_answer='{pred[:50]}...', 提取选项={extracted_option}")
        accuracy = correct / len(true_answers)
        logging.info(f"Accuracy: 正确 {correct}/{len(true_answers)} = {accuracy:.3f}")
        return accuracy
    except Exception as e:
        logging.error(f"Accuracy 计算失败: {str(e)}")
        return 0.0


def calculate_bertscore(true_answers, pred_answers, device="cpu"):
    if not true_answers or not pred_answers:
        logging.warning("真实答案或预测答案为空，返回 0.0")
        return 0.0
    try:
        logging.info(f"使用设备: {device}")
        P, R, F1 = score(pred_answers, true_answers, lang="zh", verbose=True, device=device, batch_size=16)
        f1_score = F1.mean().item()
        logging.info(f"BERTScore F1: {f1_score:.3f} (P: {P.mean().item():.3f}, R: {R.mean().item():.3f})")
        return f1_score
    except Exception as e:
        logging.error(f"BERTScore 计算失败: {str(e)}")
        return 0.0


def process_model_task(model_task_pair, results_dir):
    model, task, worker_id = model_task_pair
    logging.info(f"进程 {worker_id} 处理模型 {model} 的任务 {task}")
    model_dir = os.path.join(results_dir, model)
    json_path = os.path.join(model_dir, task, "predictions.json")
    if not os.path.exists(json_path):
        logging.warning(f"文件 {json_path} 不存在，跳过")
        return model, task, 0.0, 0

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"加载 {json_path}，共 {len(data)} 条记录")

        true_answers = []
        pred_answers = []
        error_count = 0
        for item in data:
            true_answer = item.get("answer")
            pred_answer = item.get("predicted_answer")
            if true_answer is None or pred_answer is None:
                logging.warning(f"样本 {item.get('question', '未知')} 缺少答案或预测")
                error_count += 1
                continue
            true_answers.append(str(true_answer))
            pred_answers.append(str(pred_answer))

        device = f"cuda:{worker_id % GPU_COUNT}" if GPU_COUNT > 0 else "cpu"
        score = calculate_accuracy(true_answers, pred_answers) if task == "问题类别" else calculate_bertscore(
            true_answers, pred_answers, device=device)
        score = round(score, 3)
        logging.info(f"{model} 在 {task_mapping[task]} 上的得分: {score:.3f}，错误样本数: {error_count}")
        return model, task, score, error_count

    except json.JSONDecodeError as e:
        logging.error(f"无效 JSON 文件 {json_path}: {str(e)}")
        return model, task, 0.0, len(data) if 'data' in locals() else 0
    except Exception as e:
        logging.error(f"处理 {json_path} 时出错: {str(e)}")
        return model, task, 0.0, len(data) if 'data' in locals() else 0


def calculate_scores(results_dir="./results", max_workers=None):
    models, tasks = get_models_and_tasks(results_dir)
    scores = {model: {task_mapping[task]: 0.0 for task in tasks} for model in models}
    error_counts = {model: {task_mapping[task]: 0 for task in tasks} for model in models}

    if max_workers is None:
        max_workers = min(GPU_COUNT, 4) if GPU_COUNT > 0 else min(os.cpu_count(), 4)
    logging.info(f"使用 {max_workers} 个并行进程，GPU 数量: {GPU_COUNT}")

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker,
                             initargs=(max_workers, GPU_COUNT)) as executor:
        futures = [
            executor.submit(process_model_task, (model, task, i), results_dir)
            for i, (model, task) in enumerate([(m, t) for m in models for t in tasks])
        ]
        for future in futures:
            try:
                model, task, score, error_count = future.result()
                scores[model][task_mapping[task]] = score
                error_counts[model][task_mapping[task]] = error_count
            except Exception as e:
                logging.error(f"并行任务失败: {str(e)}")

    return scores, error_counts


def save_results(scores, error_counts, output_csv="scores.csv", error_log="error_counts.csv"):
    df_scores = pd.DataFrame(scores).T
    df_scores.to_csv(output_csv, encoding="utf-8-sig")
    logging.info(f"得分已保存到 {output_csv}")

    df_errors = pd.DataFrame(error_counts).T
    df_errors.to_csv(error_log, encoding="utf-8-sig")
    logging.info(f"错误统计已保存到 {error_log}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    scores, error_counts = calculate_scores()
    save_results(scores, error_counts)
