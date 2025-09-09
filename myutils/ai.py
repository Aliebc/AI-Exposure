import json
from openai import OpenAI
import numpy as np

ai = OpenAI(
    api_key="xxx",
    base_url="http://llms-backend.axgln.net/llms/v1"
)

class LLMClient:
    def __init__(self, api_key: str, base_url: str = None):
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def run_prompt(self, system_prompt: str, user_prompt: str, model: str = "gpt-4o") -> dict:
        """
        调用 OpenAI API 执行 Prompt, 返回 JSON 对象 + token 用量
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        # 提取内容
        content = response.choices[0].message.content.strip()

        # 截取 JSON 部分
        try:
            json_str = content[content.index("{"):content.rindex("}") + 1]
            if json_str[0] != '{' or json_str[-1] != '}':
                json_str = content[content.index("["):content.rindex("]") + 1]
            print(f"Extracted JSON: {json_str}")
        except ValueError:
            json_str = "{}"  # 没有找到有效 JSON

        # 转 JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            data = {"raw": content}

        # 添加 tokens 信息
        usage_info = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        data["_usage"] = usage_info

        return data


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