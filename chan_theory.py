import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

class ChanTheory:
    def __init__(self, k_line_period: int = 5):
        """
        初始化缠论分析类
        :param k_line_period: K线周期，默认为5分钟
        """
        self.k_line_period = k_line_period
        self.min_wave_length = 3  # 最小波段长度

    def find_extreme_points(self, prices: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        寻找序列中的极值点
        :param prices: 价格序列
        :return: 极大值点和极小值点的索引列表
        """
        tops, bottoms = [], []
        n = len(prices)
        
        if n < 3:
            return tops, bottoms
            
        for i in range(1, n-1):
            # 寻找顶分型
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                tops.append(i)
            # 寻找底分型
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                bottoms.append(i)
                
        return tops, bottoms

    def identify_trend_segments(self, prices: np.ndarray, tops: List[int], bottoms: List[int]) -> List[dict]:
        """
        识别趋势段
        :param prices: 价格序列
        :param tops: 顶分型位置列表
        :param bottoms: 底分型位置列表
        :return: 趋势段列表
        """
        segments = []
        extreme_points = sorted(tops + bottoms)
        
        for i in range(len(extreme_points)-1):
            start_idx = extreme_points[i]
            end_idx = extreme_points[i+1]
            
            if start_idx in tops:
                trend_type = 'down'
            else:
                trend_type = 'up'
                
            segment = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_price': prices[start_idx],
                'end_price': prices[end_idx],
                'trend': trend_type
            }
            segments.append(segment)
            
        return segments

    def find_buy_sell_points(self, prices: np.ndarray) -> Tuple[List[dict], List[dict]]:
        """
        找出买卖点
        :param prices: 价格序列
        :return: 买点和卖点列表
        """
        tops, bottoms = self.find_extreme_points(prices)
        segments = self.identify_trend_segments(prices, tops, bottoms)
        
        buy_points = []
        sell_points = []
        
        for i in range(1, len(segments)):
            prev_segment = segments[i-1]
            curr_segment = segments[i]
            
            # 第一类买点：前一段下跌，当前上涨，且创新低
            if (prev_segment['trend'] == 'down' and 
                curr_segment['trend'] == 'up' and 
                curr_segment['start_price'] < prev_segment['end_price']):
                buy_points.append({
                    'index': curr_segment['start_idx'],
                    'price': curr_segment['start_price'],
                    'type': '一类买点'
                })
                
            # 第一类卖点：前一段上涨，当前下跌，且创新高
            elif (prev_segment['trend'] == 'up' and 
                  curr_segment['trend'] == 'down' and 
                  curr_segment['start_price'] > prev_segment['end_price']):
                sell_points.append({
                    'index': curr_segment['start_idx'],
                    'price': curr_segment['start_price'],
                    'type': '一类卖点'
                })
                
        return buy_points, sell_points

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        分析价格数据，标注买卖点
        :param df: 包含OHLCV数据的DataFrame
        :return: 添加买卖点标记的DataFrame
        """
        prices = df['close'].values
        buy_points, sell_points = self.find_buy_sell_points(prices)
        
        # 添加买卖点标记
        df['buy_signal'] = 0
        df['sell_signal'] = 0
        
        for buy in buy_points:
            df.loc[buy['index'], 'buy_signal'] = 1
            
        for sell in sell_points:
            df.loc[sell['index'], 'sell_signal'] = 1
            
        return df

def example_usage():
    # 创建示例数据
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    np.random.seed(42)
    prices = np.random.randn(100).cumsum() + 100
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.randn(100) * 0.1,
        'high': prices + np.random.randn(100) * 0.2,
        'low': prices - np.random.randn(100) * 0.2,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # 初始化缠论分析器
    chan = ChanTheory()
    
    # 分析数据
    result_df = chan.analyze(df)
    
    # 打印买卖点
    buy_points = result_df[result_df['buy_signal'] == 1]
    sell_points = result_df[result_df['sell_signal'] == 1]
    
    print("\n买点：")
    print(buy_points[['datetime', 'close']].to_string())
    print("\n卖点：")
    print(sell_points[['datetime', 'close']].to_string())

if __name__ == "__main__":
    example_usage()
