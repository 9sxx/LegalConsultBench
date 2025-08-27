import os
import json
import logging
import time
from datetime import datetime
import ollama
from concurrent.futures import ThreadPoolExecutor
import psutil
import subprocess


log_dir = os.path.join("results", "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"run_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)


def get_ollama_models(max_param_size=80.0):
    try:
        response = ollama.list()
        logging.info(f"API 响应: {response}")
        models = response.get("models", [])
        if not models:
            logging.warning("模型列表为空")
            return []

        model_info = []
        for model in models:
            model_name = model.get("name") or model.get("model")
            if not model_name:
                logging.warning(f"模型 {model} 缺少有效名称，跳过")
                continue

            details = model.get("details", {})
            param_size_str = details.get("parameter_size", "0B")
            try:
                param_size = float(param_size_str.replace('B', ''))
            except ValueError:
                logging.warning(f"无法解析 {model_name} 的参数大小 '{param_size_str}'，设为默认值 0")
                param_size = 0.0

            if param_size <= max_param_size:
                model_info.append((model_name, param_size))
            else:
                logging.warning(f"模型 {model_name} 参数大小 {param_size}B 超过阈值 {max_param_size}B，跳过")

        if not model_info:
            logging.warning("未找到有效模型信息")
            return []

        model_info.sort(key=lambda x: x[1])
        model_names = [info[0] for info in model_info]
        logging.info(f"按显存从小到大排序的模型列表: {model_names}")
        return model_names
    except Exception as e:
        logging.error(f"无法获取模型列表: {str(e)}")
        return []


def restart_ollama():
    logging.warning("尝试重启 Ollama 服务...")
    try:
        subprocess.run(["pkill", "ollama"], check=True)
        time.sleep(2)
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)
        logging.info("Ollama 服务重启成功")
    except Exception as e:
        logging.error(f"Ollama 服务重启失败: {str(e)}")


def generate_batch(model_name, prompts, samples):
    predictions = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_sample = {executor.submit(generate_response, model_name, prompt, sample): sample for prompt, sample in
                            zip(prompts, samples)}
        for future in future_to_sample:
            sample = future_to_sample[future]
            try:
                prediction = future.result(timeout=60)
                predictions.append(prediction)
            except Exception as e:
                logging.warning(f"模型 {model_name} 在样本 '{sample.get('question', '')}' 上超时或失败: {str(e)}")
                prediction = sample.copy()
                prediction["predicted_answer"] = "推理超时或失败"
                predictions.append(prediction)
    return predictions


def generate_response(model_name, prompt, sample):
    prediction = sample.copy()
    try:
        response = ollama.generate(model=model_name, prompt=prompt, stream=False)
        predicted_answer = response["response"]
        logging.info(
            f"模型 {model_name} 成功生成回答: {predicted_answer if len(predicted_answer) <= 50 else predicted_answer[:50] + '...'}")
    except Exception as e:
        logging.warning(f"模型 {model_name} 在问题 '{prompt}' 上推理失败，可能因显存不足: {str(e)}")
        predicted_answer = "推理失败（可能显存不足）"
    prediction["predicted_answer"] = predicted_answer
    return prediction


datasets = ["法条引用.json", "解决方案.json", "问题类别.json", "争议焦点.json", "整体回复.json"]

output_dir = "./results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

models = get_ollama_models(max_param_size=80.0)
if not models:
    logging.error("未找到任何模型，程序退出。")
    exit(1)

for model_name in models:
    logging.info(f"开始处理模型: {model_name}")
    model_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for dataset_path in datasets:
        if not os.path.exists(dataset_path):
            logging.warning(f"数据集文件 {dataset_path} 不存在，跳过...")
            continue
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        dataset_dir = os.path.join(model_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"成功加载数据集: {dataset_path}")
            if not data:
                logging.warning(f"数据集 {dataset_path} 为空，跳过...")
                continue
        except FileNotFoundError:
            logging.warning(f"数据集文件 {dataset_path} 不存在，跳过...")
            continue
        except json.JSONDecodeError as e:
            logging.error(f"解析 {dataset_path} 失败: {str(e)}")
            continue

        predictions = []

        prompts = [f"{sample.get('instruction', '')}\n{sample.get('question', '')}" if sample.get('instruction',
                                                                                                  '') else sample.get(
            'question', '') for sample in data]

        predictions_file = os.path.join(dataset_dir, "predictions.json")
        if os.path.exists(predictions_file):
            logging.info(f"文件 {predictions_file} 已存在，跳过处理")
            continue

        predictions = generate_batch(model_name, prompts, data)

        try:
            with open(predictions_file, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, ensure_ascii=False, indent=4)
            logging.info(f"成功保存 {model_name} 在 {dataset_name} 上的回答到 {predictions_file}")
        except Exception as e:
            logging.error(f"保存 {predictions_file} 失败: {str(e)}")

    memory = psutil.virtual_memory()
    if memory.percent > 90:
        logging.warning(f"内存使用率 {memory.percent}% 过高，考虑重启 Ollama")
        restart_ollama()

    logging.info(f"模型 {model_name} 处理完成")

logging.info("所有模型的回答已保存完成！")
