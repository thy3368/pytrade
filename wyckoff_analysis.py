from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class WyckoffAnalyzer:
    def __init__(self):
        self.client = Client()
        self.btc_data = None
    
    def get_market_data(self, symbol="TRUMPUSDT", interval="1h", limit=500):
        """获取市场数据"""
        # 获取目标币种K线数据
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        # 同时获取BTC数据作为参考
        btc_klines = self.client.get_klines(
            symbol="BTCUSDT",
            interval=interval,
            limit=limit
        )
        
        # 处理BTC数据
        self.btc_data = pd.DataFrame(btc_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 
            'volume', 'close_time', 'quote_volume', 'trades_count',
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        self.btc_data['timestamp'] = pd.to_datetime(self.btc_data['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close']:
            self.btc_data[col] = self.btc_data[col].astype(float)
        
        # 转换为DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 
            'volume', 'close_time', 'quote_volume', 'trades_count',
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        
        # 数据处理
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_volume', 'trades_count']:
            df[col] = df[col].astype(float)
        
        # 计算主动买卖指标
        df['taker_sell_volume'] = df['volume'] - df['taker_buy_volume']
        df['buy_ratio'] = (df['taker_buy_volume'] / df['volume']) * 100
        df['sell_ratio'] = (df['taker_sell_volume'] / df['volume']) * 100
        
        # 计算主动买卖笔数
        df['buy_trades'] = df['trades_count'] * (df['taker_buy_volume'] / df['volume'])
        df['sell_trades'] = df['trades_count'] * (df['taker_sell_volume'] / df['volume'])
        df['avg_buy_size'] = df['taker_buy_volume'] / df['buy_trades']
        df['avg_sell_size'] = df['taker_sell_volume'] / df['sell_trades']
        
        return df

    def analyze_wyckoff_phase(self, df):
        """威科夫阶段分析"""
        # 计算技术指标
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['price_ma20'] = df['close'].rolling(window=20).mean()
        df['price_ma50'] = df['close'].rolling(window=50).mean()
        df['buy_ratio_ma20'] = df['buy_ratio'].rolling(window=20).mean()
        
        # 计算价格趋势
        df['trend'] = np.where(df['price_ma20'] > df['price_ma50'], 'up', 'down')
        
        # 计算成交量特征
        df['volume_trend'] = df['volume'] > df['volume_ma20']
        df['buy_strength'] = df['buy_ratio'] > 50  # 主动买入占优
        
        # 获取最新状态
        latest = df.iloc[-1]
        recent = df.iloc[-20:]
        
        # 分析威科夫阶段
        phase = self._determine_phase(recent, latest)
        
        return phase, df

    def plot_analysis(self, df, phase):
        """绘制分析图表"""
        fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('价格走势', 'BTC走势', '成交量分析', '主动买卖比例', '交易笔数分析'),
                           row_heights=[0.3, 0.2, 0.2, 0.15, 0.15])

        # 添加目标币种K线图
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='目标币种'
        ), row=1, col=1)

        # 添加BTC K线图
        if self.btc_data is not None:
            fig.add_trace(go.Candlestick(
                x=self.btc_data['timestamp'],
                open=self.btc_data['open'],
                high=self.btc_data['high'],
                low=self.btc_data['low'],
                close=self.btc_data['close'],
                name='BTC'
            ), row=2, col=1)

        # 添加均线
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['price_ma20'],
            name='MA20',
            line=dict(color='orange')
        ), row=1, col=1)

        # 添加成交量
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='成交量'
        ), row=3, col=1)

        # 添加主动买卖比例
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['buy_ratio'],
            name='主动买入比例',
            line=dict(color='red')
        ), row=4, col=1)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['sell_ratio'],
            name='主动卖出比例',
            line=dict(color='green')
        ), row=4, col=1)

        # 添加主动买卖笔数
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['buy_trades'],
            name='主动买入笔数',
            marker_color='red',
            opacity=0.7
        ), row=5, col=1)

        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['sell_trades'],
            name='主动卖出笔数',
            marker_color='green',
            opacity=0.7
        ), row=5, col=1)

        # 更新布局
        fig.update_layout(
            title=f'威科夫分析 - 当前阶段: {phase}',
            xaxis_title='时间',
            yaxis_title='价格',
            height=1500
        )

        return fig

    def _determine_phase(self, recent, latest):
        """确定威科夫阶段"""
        # 计算关键指标
        price_volatility = recent['close'].std() / recent['close'].mean()
        volume_trend = recent['volume'].pct_change().mean()
        price_trend = recent['close'].pct_change().mean()
        trade_size_ratio = latest['avg_buy_size'] / latest['avg_sell_size']
        
        # 阶段判断逻辑
        if price_volatility < 0.02 and volume_trend > 0:
            if trade_size_ratio > 1.2:  # 买单规模明显大于卖单
                return "积累阶段 (强势积累)"
            return "积累阶段 (Accumulation)"
        elif price_trend > 0 and volume_trend > 0:
            if trade_size_ratio > 1.5:  # 买单规模远大于卖单
                return "上升阶段 (强势上涨)"
            return "上升阶段 (Markup)"
        elif price_volatility < 0.02 and volume_trend < 0:
            if trade_size_ratio < 0.8:  # 卖单规模明显大于买单
                return "分配阶段 (强势分配)"
            return "分配阶段 (Distribution)"
        elif price_trend < 0 and volume_trend > 0:
            if trade_size_ratio < 0.5:  # 卖单规模远大于买单
                return "下降阶段 (强势下跌)"
            return "下降阶段 (Markdown)"
        else:
            return "过渡阶段 (Transition)"

    def predict_trend(self, df):
        """预测未来趋势"""
        recent = df.iloc[-20:]  # 获取最近20个周期数据
        
        # 计算关键指标
        price_trend = recent['close'].pct_change().mean()
        volume_trend = recent['volume'].pct_change().mean()
        price_volatility = recent['close'].std() / recent['close'].mean()
        
        # 计算支撑和阻力位
        support = recent['low'].min()
        resistance = recent['high'].max()
        current_price = recent['close'].iloc[-1]
        price_position = (current_price - support) / (resistance - support)
        
        # 威科夫趋势预测逻辑
        if price_trend > 0 and volume_trend > 0:
            if price_position < 0.3:
                return {
                    'direction': '↗️ 看涨',
                    'description': '处于累积阶段末期，可能即将上涨'
                }
            elif price_position > 0.7:
                return {
                    'direction': '↘️ 看跌',
                    'description': '接近阻力位，可能开始分配'
                }
        elif price_trend < 0 and volume_trend < 0:
            if price_position < 0.3:
                return {
                    'direction': '→ 盘整',
                    'description': '可能开始累积，等待确认'
                }
            elif price_position > 0.7:
                return {
                    'direction': '↘️ 看跌',
                    'description': '分配阶段，继续下跌概率大'
                }
        
        # 判断春季测试或秋季测试
        if price_volatility < 0.02:
            if volume_trend > 0:
                return {
                    'direction': '↗️ 看涨',
                    'description': '可能是春季测试，准备上涨'
                }
            else:
                return {
                    'direction': '↘️ 看跌',
                    'description': '可能是秋季测试，准备下跌'
                }
        
        return {
            'direction': '→ 观望',
            'description': '目前趋势不明确，建议等待'
        }

    def plot_analysis(self, df, phase):
        """绘制分析图表"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('价格走势', '成交量分析'),
                           row_heights=[0.7, 0.3])

        # 添加K线图
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线'
        ), row=1, col=1)

        # 添加均线
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['price_ma20'],
            name='MA20',
            line=dict(color='orange')
        ), row=1, col=1)

        # 添加成交量
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='成交量'
        ), row=2, col=1)

        # 更新布局
        fig.update_layout(
            title=f'威科夫分析 - 当前阶段: {phase}',
            xaxis_title='时间',
            yaxis_title='价格',
            height=800
        )

        return fig

def main():
    analyzer = WyckoffAnalyzer()
    
    # 获取数据
    df = analyzer.get_market_data()
    
    # 分析阶段
    phase, df = analyzer.analyze_wyckoff_phase(df)
    
    # 打印分析结果
    print(f"\n当前威科夫阶段: {phase}")
    print("\n市场特征:")
    print(f"最新价格: {df['close'].iloc[-1]:.2f}")
    print(f"20日均价: {df['price_ma20'].iloc[-1]:.2f}")
    print(f"成交量趋势: {'上升' if df['volume_trend'].iloc[-1] else '下降'}")
    
    # 绘制图表
    fig = analyzer.plot_analysis(df, phase)
    fig.show()

if __name__ == "__main__":
    main()