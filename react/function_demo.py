from openai import OpenAI
from typing import Dict, Any
import json
from wyckoff_analysis import WyckoffAnalyzer

def get_market_data(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 100) -> Dict:
    """获取市场数据并进行分析"""
    analyzer = WyckoffAnalyzer()
    df = analyzer.get_market_data(symbol, interval, limit)
    phase, _ = analyzer.analyze_wyckoff_phase(df)
    
    return {
        "current_price": float(df['close'].iloc[-1]),
        "phase": phase,
        "volume_trend": "上升" if df['volume_trend'].iloc[-1] else "下降"
    }

class TradingAssistant:
    def __init__(self):
        # self.client = OpenAI(
        #     api_key="your-api-key",
        #     base_url="https://tbnx.plus7.plus/v1"
        # )

        self.client = OpenAI(api_key="sk-XvWvy8rf8ewno0vVFmtgOMidGb5i3h1qNQmer7bE2buY6hlK",
                             base_url="https://tbnx.plus7.plus/v1")

        self.functions = [
            {
                "name": "get_market_data",
                "description": "获取指定交易对的市场数据和分析结果",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "交易对名称，例如 BTCUSDT"
                        },
                        "interval": {
                            "type": "string",
                            "description": "K线间隔：1h, 4h, 1d"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "获取的数据点数量"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        ]

    def chat(self, user_message: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "你是一个交易分析助手。当用户询问市场状态时，你需要调用 get_market_data 函数获取数据进行分析。"
            },
            {"role": "user", "content": user_message}
        ]
        
        try:
            # 直接调用函数获取数据
            function_response = get_market_data(symbol="BTCUSDT")
            
            # 将数据添加到消息中
            messages.append({
                "role": "function",
                "name": "get_market_data",
                "content": json.dumps(function_response, ensure_ascii=False)
            })
            
            # 让模型生成最终回答
            final_response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages
            )
            
            return final_response.choices[0].message.content
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return f"获取数据时发生错误: {str(e)}"
        
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            functions=self.functions,
            function_call={"name": "get_market_data"}  # 强制调用函数
        )
        
        response_message = response.choices[0].message
        
        # 处理函数调用
        if response_message.function_call:
            function_name = response_message.function_call.name
            function_args = json.loads(response_message.function_call.arguments)
            
            if function_name == "get_market_data":
                function_response = get_market_data(**function_args)
                
                # 将函数结果发送回模型进行总结
                messages.append(response_message)
                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(function_response, ensure_ascii=False)
                })
                
                final_response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages
                )
                
                return final_response.choices[0].message.content
        
        return response_message.content

def main():
    assistant = TradingAssistant()
    
    # 测试对话
    question = "请分析一下比特币现在的市场状态"
    response = assistant.chat(question)
    print(f"问题: {question}\n")
    print(f"回答: {response}")

if __name__ == "__main__":
    main()