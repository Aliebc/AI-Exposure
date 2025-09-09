import os
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from myutils.prompts import build_ai_prompt
from myutils.ai import LLMClient

# ==== 配置 ====
EXCEL_DIR = "../data/db_29_3_excel/"
EXCEL_TASK = os.path.join(EXCEL_DIR, 'Task Statements.xlsx')
df2 = pd.read_excel(EXCEL_TASK)
OUTPUT_PATH = "./output/task_ai_exposure.csv"
API_KEY = "xxx"
BASE_URL = "http://llms-backend.axgln.net/llms/v1"
MODEL = "gpt-4o"
WORKERS = 2

os.makedirs("./output", exist_ok=True)

# ==== 初始化 LLM ====
llm = LLMClient(api_key=API_KEY, base_url=BASE_URL)

# ==== 读取数据 ====
df = pd.read_excel(EXCEL_TASK)
df = df[['Task ID', 'Task']].drop_duplicates()
df['Task ID'] = df['Task ID'].astype(str)

# ==== 定义任务函数 ====
def process_row(task_id, task_content):
    max_retries = 3
    for attempt in range(max_retries):
        system_prompt, user_prompt = build_ai_prompt(task_id, task_content)
        try:
            result = llm.run_prompt(system_prompt, user_prompt, model=MODEL)

            if all(k in result for k in ['score', 'exposure', 'reason', '_usage']):
                return {
                    "Task_ID": task_id,
                    "Task": task_content,
                    "score": result["score"],
                    "exposure": result["exposure"],
                    "reason": result["reason"],
                    "_usage": json.dumps(result["_usage"], ensure_ascii=False),
                    "_model": MODEL
                }
        except Exception as e:
            print(f"[{task_id}] Error: {e} (attempt {attempt+1})")

    # 超过重试次数，返回空数据
    return {
        "Task_ID": task_id,
        "Task": task_content,
        "score": 0,
        "exposure": "ERROR",
        "reason": "",
        "_usage": "{}",
        "_model": MODEL
    }

# ==== 并行处理（带进度条） ====
results = []
with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    future_to_id = {
        executor.submit(process_row, row['Task ID'], row['Task']): row['Task ID']
        for _, row in df.iterrows()
    }

    for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Processing"):
        result = future.result()
        results.append(result)
        #print(f"Processed {result['DWA_ID']}: {result['score']} {result['exposure']}")

# ==== 保存为 CSV ====
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"Saved to {OUTPUT_PATH}")
