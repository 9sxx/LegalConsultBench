import pandas as pd
import ollama
import json
import logging
import signal
import sys
from datetime import datetime
import os

df = None
current_idx = 0

logging.basicConfig(
    filename=f"data_labeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

progress_file = "progress.txt"


def save_partial_data():
    global df, current_idx
    if df is not None:
        output_file = f"consultations_labeled_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        logging.info(f"数据已保存至：{output_file}")
        with open(progress_file, 'w') as f:
            f.write(str(current_idx))
        logging.info(f"进度已保存：当前处理到第 {current_idx + 1} 条数据。")


def load_progress():
    global current_idx
    try:
        with open(progress_file, 'r') as f:
            current_idx = int(f.read().strip())
            logging.info(f"恢复进度：从第 {current_idx + 1} 条数据开始处理。")
    except FileNotFoundError:
        logging.info("没有找到进度文件，程序将从头开始处理。")
        current_idx = 0


def signal_handler(sig, frame):
    logging.info("捕获到中断信号，正在保存已处理的数据...")
    save_partial_data()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def extract_law_and_solution(content, index):
    global df, current_idx
    try:
        logging.info(f"开始处理第 {index + 1} 条数据的法条引用与解决方案...")
        prompt = f"""
        你是一个专业的法律文本分析助手，任务是从给定的文本中提取法条引用和解决方案。

        1. 法条引用：提取所有引用的法律条款，包括法典名称、条文序号及其具体内容。
        2. 解决方案：提取针对问题提供的建议、方案或解释部分。

        输出格式：
        {{
            "法条引用": "提取的法条引用内容",
            "解决方案": "提取的解决方案内容"
        }}

        输入文本：
        \"{content}\"
        """
        response = ollama.chat(
            model='qwen2:72b',
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        result = json.loads(response['message']['content'])
        return result.get("法条引用", ""), result.get("解决方案", "")
    except Exception as e:
        logging.error(f"处理第 {index + 1} 条数据时出错: {e}")
        save_partial_data()
        return "", ""


def extract_dispute_focus(question, category, content, index):
    global df, current_idx
    try:
        logging.info(f"开始处理第 {index + 1} 条数据的争议焦点...")
        prompt = f"""
        你是一个专业的法律助手，任务是从以下信息中提取出争议焦点。

        输入信息：
        - 咨询问题: {question}
        - 问题类别: {category}
        - 内容: {content}

        请提取出该问题的争议焦点，并用一句话概括。
        输出格式：
        {{
            "争议焦点": "提取的争议焦点"
        }}
        """
        response = ollama.chat(
            model='qwen2:72b',
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        result = json.loads(response['message']['content'])
        dispute_focus = result.get("争议焦点", "")
        logging.info(f"第 {index + 1} 条争议焦点提取完成: {dispute_focus}")
        return dispute_focus
    except Exception as e:
        logging.error(f"提取第 {index + 1} 条数据的争议焦点时出错: {e}")
        save_partial_data()
        return ""


file_path = "consultations_data.csv"
try:
    logging.info("开始读取数据文件...")
    df = pd.read_csv(file_path)
    logging.info("数据文件读取成功.")
except Exception as e:
    logging.error(f"读取数据文件时出错: {e}")
    sys.exit()

required_columns = ['咨询问题', '问题类别', '内容', '回复']
if all(col in df.columns for col in required_columns):
    for col in ['法条引用', '解决方案', '争议焦点']:
        if col not in df.columns:
            df[col] = ""

    load_progress()

    total_rows = df.shape[0]
    logging.info(f"总共有 {total_rows} 条数据需要处理。")

    try:
        for pos, row in enumerate(df.iloc[current_idx:].itertuples(index=False), start=current_idx):
            law, solution = extract_law_and_solution(row.回复, pos)
            dispute_focus = extract_dispute_focus(row.咨询问题, row.问题类别, row.内容, pos)

            df.at[pos, '法条引用'] = law
            df.at[pos, '解决方案'] = solution
            df.at[pos, '争议焦点'] = dispute_focus

            current_idx += 1

            if current_idx % 5000 == 0:
                save_partial_data()

        if os.path.exists(progress_file):
            os.remove(progress_file)
            logging.info("所有数据处理完成，进度文件已删除。")

        output_file = "consultations_labeled_data.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        logging.info(f"所有数据处理完成，结果已保存至：{output_file}")

    except Exception as e:
        logging.error(f"在处理过程中发生未捕获的异常: {e}")
        save_partial_data()
        sys.exit()

else:
    logging.error("数据文件中缺少必需的列，请检查文件内容。")
