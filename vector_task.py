import os
import time
import pandas as pd
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time


# 向量化模型接口设置
from openai import OpenAI
ai = OpenAI(
    api_key="xxx",
    base_url="http://llms-backend.axgln.net/llms/v1"
)

# Step 1: 加载并准备数据
os.makedirs('./output', exist_ok=True)
EXCEL_DIR = "../data/db_29_3_excel/"
EXCEL_DWA = os.path.join(EXCEL_DIR, 'Task Statements.xlsx')
df2 = pd.read_excel(EXCEL_DWA)
TASKs = df2[['Task ID', 'Task']].drop_duplicates()
TASKs['Task ID'] = TASKs['Task ID'].astype(str)
task_table = TASKs.copy()
#task_table = TASKs.head(10)  # 只取前10行示例
print(task_table)

def get_embeddings_multithread(texts, model="text-embedding-3-large", max_workers=2):
    def process_text(i, text):
        try:
            response = ai.embeddings.create(input=text, model=model)
            return i, response.data[0].embedding
        except Exception as e:
            print(f"Error at {i}: {e}")
            return i, [0] * 1536  # fallback

    embeddings = [None] * len(texts)  # 预分配
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_text, i, text): i for i, text in enumerate(texts)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing embeddings"):
            i, emb = future.result()
            embeddings[i] = emb
            #time.sleep(0.05)  # 避免速率限制

    return embeddings

# Step 3: 执行向量化
titles = task_table['Task'].tolist()
embeddings = get_embeddings_multithread(titles, max_workers=2)


# Step 4: 合并到 DataFrame
task_table['embedding'] = embeddings

# Step 5: 保存为 Python 版本无关的格式（推荐 JSON Lines）
output_path = './output/task_vectors.jsonl'
with open(output_path, 'w', encoding='utf-8') as f:
    for _, row in task_table.iterrows():
        json_line = {
            'Task_ID': row['Task ID'],
            'Task': row['Task'],
            'embedding': row['embedding']
        }
        f.write(json.dumps(json_line) + '\n')

print(f"向量数据已保存到 {output_path}")