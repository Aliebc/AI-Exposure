import os
import time
import pandas as pd
import numpy as np
import json
from openai import OpenAI
ai = OpenAI(
    api_key="xxx",
    base_url="http://llms-backend.axgln.net/llms/v1"
)

CSV_DIR = "../data/zlzp/"

def vectorize_texts(text, model="text-embedding-3-large"):
    try:
        response = ai.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error during embedding: {e}")
        return [0] * 1536

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0.0
    return np.dot(v1, v2) / norm_product

df = pd.read_json('./output/dwa_vectors.jsonl', lines=True)
print(df.head())

job_table = pd.read_csv(os.path.join(CSV_DIR, 'top_50_job_content_samplepart0d.csv'))
job_table = job_table[~job_table['intro'].isna()]
print(job_table.head())

for i, row in job_table.iterrows():
    mixed_text = f"{row['title']}: {row['intro']}"
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
    df_new = df_new.head(10)  # 取前10个最相
    with open(f'./output/job_dwa_{mid}.txt', 'w', encoding='utf-8') as f:
        f.write(f"Job ID: {mid}\n")
        f.write(f"Job Title: {row['title']}\n")
        f.write(f"Job Intro: {row['intro']}\n")
        f.write("Top 10 DWAs:\n")
        for _, new_row in df_new.iterrows():
            f.write(f"{new_row['DWA_ID']}\t{new_row['DWA_Title']}\t{new_row['similarity']}\n")
    print(f"Processed job {i+1}/{len(job_table)}: {mid}")