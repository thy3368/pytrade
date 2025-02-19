from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from chan_theory import ChanTheory

class TrumpTradeAnalyzer:
    def __init__(self):
        # 初始化 Binance 客户端
        self.binance_client = Client()
        self.symbol = "TRUMPUSDT"
        self.chan = ChanTheory(k_line_period=5)  # 5分钟K线周期

    def get_klines(self, interval="5m", lookback_hours=24):
        """获取K线数据"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        klines = self.binance_client.get_historical_klines(
            self.symbol,
            interval,
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # 转换为DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                         'close_time', 'quote_volume', 'trades_count',
                                         'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'])
        
        # 数据处理
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df

    def analyze_market(self, interval="5m", lookback_hours=24):
        """分析市场数据"""
        df = self.get_klines(interval, lookback_hours)
        
        # 计算基本统计
        current_price = float(df['close'].iloc[-1])
        price_change = ((current_price - float(df['open'].iloc[0])) / float(df['open'].iloc[0])) * 100
        high_price = float(df['high'].max())
        low_price = float(df['low'].min())
        
        # 计算成交量统计
        avg_volume = df['volume'].mean()
        total_volume = df['volume'].sum()
        
        # 使用缠论分析
        analysis_df = self.chan.analyze(df)
        buy_points = analysis_df[analysis_df['buy_signal'] == 1]
        sell_points = analysis_df[analysis_df['sell_signal'] == 1]
        
        # 生成分析报告
        report = {
            'current_price': current_price,
            'price_change_percent': price_change,
            'high_price': high_price,
            'low_price': low_price,
            'avg_volume': avg_volume,
            'total_volume': total_volume,
            'latest_buy_signals': len(buy_points[buy_points['timestamp'] > df['timestamp'].iloc[-10]]),
            'latest_sell_signals': len(sell_points[sell_points['timestamp'] > df['timestamp'].iloc[-10]]),
            'timestamp': datetime.now()
        }
        
        return report, analysis_df

    def print_analysis(self, report):
        """打印分析报告"""
        print(f"\n=== TRUMP/USDT 市场分析 ===")
        print(f"分析时间: {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n当前价格: ${report['current_price']:.4f}")
        print(f"24小时涨跌幅: {report['price_change_percent']:.2f}%")
        print(f"最高价: ${report['high_price']:.4f}")
        print(f"最低价: ${report['low_price']:.4f}")
        print(f"\n平均成交量: {report['avg_volume']:.2f}")
        print(f"总成交量: {report['total_volume']:.2f}")
        print(f"\n最近10根K线买入信号: {report['latest_buy_signals']}")
        print(f"最近10根K线卖出信号: {report['latest_sell_signals']}")

def main():
    analyzer = TrumpTradeAnalyzer()
    
    # 获取5分钟K线的24小时数据并分析
    report, analysis_df = analyzer.analyze_market(interval="5m", lookback_hours=24)
    
    # 打印分析报告
    analyzer.print_analysis(report)
    
    # 保存分析数据到CSV（可选）
    analysis_df.to_csv('trump_analysis.csv', index=False)

if __name__ == "__main__":
    main()
