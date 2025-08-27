import pandas as pd
import re
import numpy as np

file_path = "datas/consultations_data.csv"

df = pd.read_csv(file_path)
df = df[['咨询问题', '问题类别', '内容', '回复']]


def clean_content(content):
    if not isinstance(content, str):
        return ""
    match = re.search(
        r'您好！欢迎关注中国法律服务网。\s*根据您所表达的需求，我们为您提供如下信息：(.*?)\n中国法律服务网平台为您提供以上信息，仅供您参考。如有疑问，欢迎进一步咨询。感谢您对中国法律服务网的关注和支持！',
        content, re.DOTALL)
    return match.group(1).strip() if match else ""


df['回复'] = df['回复'].apply(clean_content)
df = df.replace("", np.nan)
df = df.dropna()


def contains_valid_characters(value):
    if isinstance(value, str):
        return bool(re.search(r'[a-zA-Z\u4e00-\u9fa5]', value))
    return False


for column in df.columns:
    df = df[df[column].apply(contains_valid_characters)]

df = df[df['问题类别'] != '其他']

output_file = "consultations_data.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")
