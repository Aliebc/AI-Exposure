def build_prompt(id: str, title: str) -> str:
    """
    根据 DWA 生成 AI 替代性评估 Prompt（纯文本，要求 JSON 输出）
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
