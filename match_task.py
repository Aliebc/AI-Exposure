import os
import pandas as pd
from myutils.ai import vectorize_texts, cosine_similarity
from myutils.prompts import build_split_prompt
from myutils.ai import LLMClient

API_KEY = "xxx"
BASE_URL = "http://llms-backend.axgln.net/llms/v1"
llm = LLMClient(api_key=API_KEY, base_url=BASE_URL)

CSV_DIR = "../data/zlzp/"

df = pd.read_parquet('../data/eva/task_vectors.parquet')
print(df.head())

os.makedirs('./output', exist_ok=True)

job_table = pd.read_csv(os.path.join(CSV_DIR, 'top_50_job_content_samplepart0d.csv'))
job_table = job_table[~job_table['intro'].isna()]
print(job_table.head())

for i, row in job_table.iterrows():
    mixed_text = f"{row['title']}: {row['intro']}"
    print(mixed_text)
    s, u = build_split_prompt(mixed_text)
    d = llm.run_prompt(s, u)
    print(d)
    exit(0)
    job_vector = vectorize_texts(mixed_text)
    mid = row['mid']
    # 构造新表  
    df_new = df.copy()
    df_new['similarity'] = 0.0
    for j, dwa_row in df.iterrows():
        dwa_vector = dwa_row['embedding']
        similarity = cosine_similarity(job_vector, dwa_vector)
        df_new.at[j, 'similarity'] = similarity
    df_new = df_new.sort_values(by='similarity', ascending=False)
    df_new = df_new.head(10)  # 取前10
    with open(f'./output/job_task_{mid}.txt', 'w', encoding='utf-8') as f:
        f.write(f"Job ID: {mid}\n")
        f.write(f"Job Title: {row['title']}\n")
        f.write(f"Job Intro: {row['intro']}\n")
        f.write("Top 10 Tasks:\n")
        for _, new_row in df_new.iterrows():
            f.write(f"{new_row['Task_ID']}\t{new_row['Task']}\t{new_row['similarity']}\n")
    print(f"Processed job {i+1}/{len(job_table)}: {mid}")