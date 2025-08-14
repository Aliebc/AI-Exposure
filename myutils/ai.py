import json
from openai import OpenAI

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
