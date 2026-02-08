import json
import re

def parse_model_response(response_text: str):
    """
    尝试将模型的文本输出清洗并转换为Python对象(List/Dict)。
    如果失败，返回原始字符串。
    """
    # 1. 尝试直接解析
    try:
        return json.loads(response_text)
    except:
        pass

    # 2. 尝试提取 Markdown 代码块中的 JSON
    try:
        pattern = r"```json(.*?)```"
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            return json.loads(json_str)
    except:
        pass

    # 3. 尝试提取最外层 [] 或 {}
    try:
        pattern = r"(\[.*\]|\{.*\})"
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            return json.loads(json_str)
    except:
        pass

    # 4. 无法解析，返回原始文本，但做简单的按行分割处理
    return {"raw_text": response_text, "parse_error": "Failed to parse JSON"}