from typing import List, Tuple

def build_ai_prompt(id: str, title: str) -> Tuple[str, str]:
    """
    根据 DWA 生成 AI 替代性评估 Prompt(纯文本, 要求 JSON 输出）
    """
    system_instructions = """
You are an expert in labor economics and AI capabilities assessment.
Your task is to evaluate the AI replaceability of a given work task,
based on the following Exposure Rubric:

- No exposure (E0):
  - Using the described LLM results in no or minimal reduction in the time required
    to complete the task while maintaining equivalent quality, OR
  - Using the described LLM results in a decrease in the quality of the task output.

- Direct exposure (E1):
  - Using the described LLM via ChatGPT or the OpenAI playground can decrease the time
    required to complete the task by at least half (50%).

- LLM+ exposure (E2):
  - Access to the described LLM alone would not reduce the time by at least half, BUT
  - Additional software could be built on top of the LLM that would reduce the time
    with quality by at least half. Access to image generation systems is counted here.
    """

    user_prompt = f"""
Task Description: {title} (DWA ID: {id})

Return your answer in JSON with the following keys:
- "score": integer 0-100, AI Replaceability Score
- "exposure": one of "E0", "E1", "E2"
- "reason": short reasoning in 1-2 sentences
    """

    return system_instructions.strip(), user_prompt.strip()

def build_split_prompt(content: str) -> Tuple[str, str]:
    """
    根据工作内容切分 Prompt(纯文本, 要求 JSON 输出）
    输入：完整的职位描述 JD 文本
    输出：system prompt 与 user prompt
    """

    system_instructions = """
You are an expert in human resources and job analysis.
Your task is to split the given job description (JD) into two categories:

1. 工作内容 (Responsibilities / Duties)
2. 任职要求 (Qualifications / Requirements)

Rules:
- You must keep the **original wording** from the JD without rewriting.
- Output must cover **all parts of the JD**.
- Return the result in **JSON array** format, where each element is an object with keys:
  - "category": either "工作内容" or "任职要求"
  - "text": the exact extracted sentence(s) from the original JD
    """

    user_prompt = f"""
Job Description (JD):

{content}

Now split the text strictly into 工作内容 and 任职要求.
Return your answer in JSON array format as specified.
    """

    return system_instructions.strip(), user_prompt.strip()