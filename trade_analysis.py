import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from binance.client import Client
from datetime import datetime, timedelta

def analyze_buy_initiative(df):
    """
    分析交易的主动买入情况
    df 需要包含以下列：
    - timestamp: 时间戳
    - price: 成交价格
    - volume: 成交量
    - bid: 买一价
    - ask: 卖一价
    """
    # 计算主动买入指标
    df['is_buy_initiative'] = df['price'] >= df['ask']
    df['is_sell_initiative'] = df['price'] <= df['bid']
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('价格与主动买入分析', '交易量分析'),
        row_heights=[0.7, 0.3]
    )

    # 添加价格线
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['price'],
            name='成交价',
            line=dict(color='gray')
        ),
        row=1, col=1
    )

    # 添加主动买入点
    buy_df = df[df['is_buy_initiative']]
    fig.add_trace(
        go.Scatter(
            x=buy_df['timestamp'],
            y=buy_df['price'],
            mode='markers',
            name='主动买入',
            marker=dict(color='red', size=8)
        ),
        row=1, col=1
    )

    # 添加主动卖出点
    sell_df = df[df['is_sell_initiative']]
    fig.add_trace(
        go.Scatter(
            x=sell_df['timestamp'],
            y=sell_df['price'],
            mode='markers',
            name='主动卖出',
            marker=dict(color='green', size=8)
        ),
        row=1, col=1
    )

    # 添加交易量柱状图
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='交易量',
            marker_color='lightgray'
        ),
        row=2, col=1
    )

    # 更新布局
    fig.update_layout(
        title='交易主动性分析',
        xaxis_title='时间',
        yaxis_title='价格',
        height=800,
        showlegend=True
    )

    return fig

def get_binance_trades(symbol, start_time, end_time=None):
    """
    获取币安交易数据
    """
    client = Client()
    end_time = end_time or datetime.now()
    
    # 获取K线数据作为买卖价格参考
    klines = client.get_historical_klines(
        symbol,
        Client.KLINE_INTERVAL_1MINUTE,
        start_time.strftime("%Y-%m-%d %H:%M:%S"),
        end_time.strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # 获取交易数据
    trades = client.get_aggregate_trades(
        symbol=symbol,
        startTime=int(start_time.timestamp() * 1000),
        endTime=int(end_time.timestamp() * 1000)
    )
    
    # Check if we got any data
    if not trades or not klines:
        raise ValueError(f"No data received for {symbol} between {start_time} and {end_time}")
    
    # 处理交易数据
    trade_data = []
    for trade in trades:
        trade_data.append({
            'timestamp': pd.to_datetime(trade['T'], unit='ms'),
            'price': float(trade['p']),
            'volume': float(trade['q']),
            'is_buyer_maker': trade['m']
        })
    
    df = pd.DataFrame(trade_data)
    
    # 处理K线数据用于买卖价格
    kline_data = []
    for k in klines:
        kline_data.append({
            'timestamp': pd.to_datetime(k[0], unit='ms'),
            'bid': float(k[3]),  # 最低价作为买一价
            'ask': float(k[2])   # 最高价作为卖一价
        })
    
    kline_df = pd.DataFrame(kline_data)
    
    # Debug prints
    print(f"Trade data shape: {df.shape}")
    print(f"Kline data shape: {kline_df.shape}")
    print(f"Trade data columns: {df.columns}")
    print(f"Kline data columns: {kline_df.columns}")
    
    # Check if DataFrames are empty
    if df.empty or kline_df.empty:
        raise ValueError("Empty DataFrame(s) created")
    
    # Sort and merge
    df = df.sort_values('timestamp')
    kline_df = kline_df.sort_values('timestamp')
    
    df = pd.merge_asof(
        df,
        kline_df,
        on='timestamp',
        direction='nearest'
    )
    
    return df

# 示例使用
if __name__ == "__main__":
    try:
        # 获取最近1小时的BTC/USDT交易数据
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        print(f"Fetching data from {start_time} to {end_time}")
        df = get_binance_trades('BTCUSDT', start_time, end_time)
        
        # 创建并显示图表
        fig = analyze_buy_initiative(df)
        fig.show()
    except Exception as e:
        print(f"Error occurred: {e}")