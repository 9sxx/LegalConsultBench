import pandas as pd

file_path = 'consultations_labeled_data.csv'
data = pd.read_csv(file_path)

data_cleaned = data.dropna()

data_cleaned = data_cleaned[data_cleaned['法条引用'].str.len() >= 20]

data_cleaned = data_cleaned[~data_cleaned['法条引用'].str.match(r'^\[.*\]$')]

data_cleaned = data_cleaned.drop_duplicates()

data_cleaned['回复'] = data_cleaned['回复'].str.replace(
    "根据您所表达的需求，我们为您提供如下信息：", "", regex=False
).str.strip()

output_file_path = 'cleaned_consultations_labeled_data.xlsx'
data_cleaned.to_excel(output_file_path, index=False)
