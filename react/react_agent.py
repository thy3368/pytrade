from openai import OpenAI
from typing import List, Dict

class ReActAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.lkeap.cloud.tencent.com/v1"
        )
        self.thought_history: List[Dict] = []
    
    def think_and_act(self, query: str) -> str:
        # 构建提示词
        prompt = f"""请按以下格式思考并回答问题:
问题: {query}

思考: 让我逐步分析这个问题
行动: 基于分析采取的行动
观察: 行动的结果
结论: 最终答案

请严格按照上述格式回复。"""
        
        # 调用 DeepSeek API
        response = self.client.chat.completions.create(
            model="deepseek-v1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content

def main():
    # 初始化 agent
    api_key = "sk-aYPrL01TCkSj1gpdkqDC5iCovqot6Amw569oYNIXzk3ay29M"  # 替换为你的 OpenAI API key
    agent = ReActAgent(api_key)
    
    # 测试问题
    question = "如何分析比特币的价格趋势？"
    result = agent.think_and_act(question)
    print(result)

if __name__ == "__main__":
    main()