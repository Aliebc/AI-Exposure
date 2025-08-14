import os
import time
import pandas as pd
import numpy as np
import json

# 向量化模型接口设置
from openai import OpenAI
ai = OpenAI(
    api_key="xxx",
    base_url="http://llms-backend.axgln.net/llms/v1"
)

# Step 1: 加载并准备数据
os.makedirs('./output', exist_ok=True)
EXCEL_DIR = "../data/db_29_3_excel/"
EXCEL_DWA = os.path.join(EXCEL_DIR, 'DWA Reference.xlsx')
df2 = pd.read_excel(EXCEL_DWA)
DWAs = df2[['DWA ID', 'DWA Title']].drop_duplicates()
DWAs['DWA ID'] = DWAs['DWA ID'].astype(str)
dwa_table = DWAs.copy()
#dwa_table = DWAs.head(10)  # 只取前10行示例
print(dwa_table)

# Step 2: 向量化函数
def get_embeddings(texts, model="text-embedding-3-large"):
    embeddings = []
    for i, text in enumerate(texts):
        try:
            response = ai.embeddings.create(input=text, model=model)
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"Error at {i}: {e}")
            embeddings.append([0]*1536)  # fallback
        print(f"Processed {i+1}/{len(texts)}: {text}")
        time.sleep(0.05)  # avoid rate limit
    return embeddings

# Step 3: 执行向量化
titles = dwa_table['DWA Title'].tolist()
embeddings = get_embeddings(titles)

# Step 4: 合并到 DataFrame
dwa_table['embedding'] = embeddings

# Step 5: 保存为 Python 版本无关的格式（推荐 JSON Lines）
output_path = './output/dwa_vectors.jsonl'
with open(output_path, 'w', encoding='utf-8') as f:
    for _, row in dwa_table.iterrows():
        json_line = {
            'DWA_ID': row['DWA ID'],
            'DWA_Title': row['DWA Title'],
            'embedding': row['embedding']
        }
        f.write(json.dumps(json_line) + '\n')

print(f"向量数据已保存到 {output_path}")

import pandas as pd
df = pd.read_json('./output/dwa_vectors.jsonl', lines=True)
print(df.head())
df.to_parquet('./output/dwa_vectors.parquet', index=False, compression='gzip')