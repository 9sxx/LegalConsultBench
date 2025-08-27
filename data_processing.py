import pandas as pd
import json
import random

excel_file = 'sampled_data_process.xlsx'
df = pd.read_excel(excel_file)

required_columns = ["咨询问题", "问题类别", "内容", "回复", "法条引用", "解决方案", "争议焦点"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"缺少必要的列: {col}")

all_categories = df["问题类别"].dropna().unique().tolist()


def create_question_category(row, all_categories):
    instruction = "阅读以下咨询问题和内容，并从选项A、B、C、D中选择最合适的问题类别。"
    question = f"{row['咨询问题']}：{row['内容']}\n\n请选择以下哪个选项最适合该问题类别：\n"
    correct_answer = row['问题类别']

    wrong_categories = [cat for cat in all_categories if cat != correct_answer]
    wrong_answers = random.sample(wrong_categories, k=min(3, len(wrong_categories)))
    options = wrong_answers + [correct_answer]
    random.shuffle(options)

    option_labels = ['A', 'B', 'C', 'D']
    labeled_options = {label: option for label, option in zip(option_labels, options)}

    for label in labeled_options:
        question += f"{label}. {labeled_options[label]}\n"

    correct_label = next(label for label, option in labeled_options.items() if option == correct_answer)

    answer = correct_label

    return {
        "instruction": instruction,
        "question": question.strip(),
        "answer": answer
    }


def create_dispute_focus_entry(row):
    instruction = "阅读以下咨询问题和内容，并概括该案例的主要争议焦点。"
    question = f"{row['咨询问题']}：{row['内容']}"
    answer = row["争议焦点"]
    return {
        "instruction": instruction,
        "question": question,
        "answer": answer
    }


def create_legal_reference_entry(row):
    instruction = "阅读以下咨询问题和内容，并列出相关的法律条文引用。"
    question = f"{row['咨询问题']}：{row['内容']}"
    answer = row["法条引用"]
    return {
        "instruction": instruction,
        "question": question,
        "answer": answer
    }


def create_solution_entry(row):
    instruction = "阅读以下咨询问题和内容，并提供相应的解决方案。"
    question = f"{row['咨询问题']}：{row['内容']}"
    answer = row["解决方案"]
    return {
        "instruction": instruction,
        "question": question,
        "answer": answer
    }


def create_overall_reply_entry(row):
    instruction = "阅读以下咨询问题和内容，并提供整体的回复。"
    question = f"{row['咨询问题']}：{row['内容']}"
    answer = row["回复"]
    return {
        "instruction": instruction,
        "question": question,
        "answer": answer
    }


question_category_list = []
dispute_focus_list = []
legal_reference_list = []
solution_list = []
overall_reply_list = []

for index, row in df.iterrows():
    qc_entry = create_question_category(row, all_categories)
    question_category_list.append(qc_entry)

    df_entry = create_dispute_focus_entry(row)
    dispute_focus_list.append(df_entry)

    lr_entry = create_legal_reference_entry(row)
    legal_reference_list.append(lr_entry)

    sl_entry = create_solution_entry(row)
    solution_list.append(sl_entry)

    or_entry = create_overall_reply_entry(row)
    overall_reply_list.append(or_entry)

json_files = {
    "问题类别.json": question_category_list,
    "争议焦点.json": dispute_focus_list,
    "法条引用.json": legal_reference_list,
    "解决方案.json": solution_list,
    "整体回复.json": overall_reply_list
}

for filename, data in json_files.items():
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
