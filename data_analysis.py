import pandas as pd

file_path = 'cleaned_consultations_labeled_data.xlsx'
data = pd.read_excel(file_path)

category_counts = data['问题类别'].value_counts()
categories_over_100 = category_counts[category_counts > 100].index

filtered_data = data[data['问题类别'].isin(categories_over_100)]

sampled_data = filtered_data.groupby('问题类别').apply(lambda x: x.sample(n=100, random_state=42)).reset_index(drop=True)

output_file_path = 'sampled_data.xlsx'
sampled_data.to_excel(output_file_path, index=False)

