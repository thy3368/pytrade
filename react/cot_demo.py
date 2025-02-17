from openai import OpenAI
from typing import Dict

class ChainOfThoughtAnalyzer:
    def __init__(self):
        self.client = OpenAI(
            api_key="sk-XvWvy8rf8ewno0vVFmtgOMidGb5i3h1qNQmer7bE2buY6hlK",
            base_url="https://tbnx.plus7.plus/v1"
        )

    def analyze(self, market_data: Dict) -> str:
        prompt = f"""请按照以下思维步骤分析市场数据：

市场数据:
- 当前价格: {market_data['price']}
- 24h涨跌幅: {market_data['change_24h']}%
- 交易量变化: {market_data['volume_change']}%

思维步骤:
1) 首先，分析价格趋势
2) 然后，评估交易量变化
3) 接着，结合价格和交易量判断市场强弱
4) 最后，给出交易建议

请按照上述步骤进行分析，每个步骤都要明确说明你的推理过程。
"""
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content

def main():
    # 示例数据
    market_data = {
        "price": 43000,
        "change_24h": 2.5,
        "volume_change": 15
    }
    
    analyzer = ChainOfThoughtAnalyzer()
    analysis = analyzer.analyze(market_data)
    
    print("市场分析结果:")
    print(analysis)

if __name__ == "__main__":
    main()