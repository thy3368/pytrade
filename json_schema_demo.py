import json

from openai import OpenAI

# 设置 API key
# 
client = OpenAI(
    api_key="sk-XvWvy8rf8ewno0vVFmtgOMidGb5i3h1qNQmer7bE2buY6hlK",
    base_url="https://tbnx.plus7.plus/v1"
)

# 定义一个简单的 JSON Schema
PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "hobbies": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["name", "age", "hobbies"]
}

def get_structured_data(description):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": f"你是一个帮助生成结构化数据的助手。请根据用户的描述，生成符合以下JSON Schema的数据：\n{json.dumps(PERSON_SCHEMA, indent=2, ensure_ascii=False)}\n只需要返回JSON数据，不要有其他说明文字。"
                },
                {
                    "role": "user",
                    "content": description
                }
            ]
        )
        
        content = response.choices[0].message.content.strip()
        # 处理可能的 markdown 格式
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
            
        return json.loads(content.strip())
    except Exception as e:
        return {
            "error": str(e),
            "raw_response": response.choices[0].message.content if 'response' in locals() else None
        }

def main():
    # 测试描述
    description = "小明是一个23岁的年轻人，他喜欢打篮球、看电影和编程。"
    
    print("输入描述:", description)
    result = get_structured_data(description)
    print("\n生成的JSON数据:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
