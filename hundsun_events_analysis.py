import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

# 设置显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def get_stock_data(stock_code='600570', start_date='20230101'):
    """获取股票数据"""
    df = ak.stock_zh_a_hist(symbol=stock_code, 
                           start_date=start_date,
                           adjust="qfq")
    
    df = df.rename(columns={
        '日期': 'Date',
        '收盘': 'Close',
        '成交量': 'Volume'
    })
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def get_key_events():
    """定义关键事件"""
    events = [
        # 财报
        {
            'date': '2023-03-15',
            'event': '2022年报',
            'detail': '+22.37%',
            'type': '财报',
            'impact': '+'
        },
        {
            'date': '2023-08-30',
            'event': '半年报',
            'detail': '+25.12%',
            'type': '财报',
            'impact': '+'
        },
        {
            'date': '2024-01-10',
            'event': '业绩预告',
            'detail': '+50-70%',
            'type': '财报',
            'impact': '+'
        },
        
        # 合作
        {
            'date': '2023-09-20',
            'event': '投资',
            'detail': '天阙科技',
            'type': '战略',
            'impact': '+'
        },
        {
            'date': '2023-11-15',
            'event': '合作',
            'detail': '华为',
            'type': '战略',
            'impact': '+'
        },
        
        # 科技
        {
            'date': '2023-04-15',
            'event': 'AI',
            'detail': '智能投顾',
            'type': '科技',
            'impact': '+'
        },
        {
            'date': '2023-07-20',
            'event': 'DCEP',
            'detail': '数字人民币',
            'type': '科技',
            'impact': '+'
        },
        {
            'date': '2023-10-25',
            'event': '区块链',
            'detail': '清算系统',
            'type': '科技',
            'impact': '+'
        },
        
        # 新闻
        {
            'date': '2023-05-10',
            'event': '监管',
            'detail': '政策收紧',
            'type': '新闻',
            'impact': '-'
        },
        {
            'date': '2023-06-18',
            'event': '竞争',
            'detail': '新进入者',
            'type': '新闻',
            'impact': '-'
        },
        {
            'date': '2023-12-05',
            'event': '扩张',
            'detail': '东南亚',
            'type': '新闻',
            'impact': '+'
        }
    ]
    return pd.DataFrame(events)

def analyze_price_impact(df, event_date, window=10):
    """分析事件前后的价格变化"""
    event_date = pd.to_datetime(event_date)
    if event_date not in df.index:
        nearest_date = df.index[df.index > event_date][0]
        event_date = nearest_date
    
    pre_price = df.loc[:event_date].iloc[-window:]['Close']
    post_price = df.loc[event_date:].iloc[:window]['Close']
    
    pre_change = (pre_price.iloc[-1] - pre_price.iloc[0]) / pre_price.iloc[0] * 100
    post_change = (post_price.iloc[-1] - post_price.iloc[0]) / post_price.iloc[0] * 100
    
    return pre_change, post_change

def plot_stock_events():
    """绘制股票走势和事件分析图"""
    df = get_stock_data()
    events = get_key_events()
    
    # 创建图形
    fig = plt.figure(figsize=(20, 16))
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # 1. 主图：股价走势和事件标记
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['Close'], label='Price', linewidth=2)
    
    # 添加事件标记
    colors = {
        '财报': 'red',
        '战略': 'blue',
        '科技': 'purple',
        '新闻': 'orange'
    }
    
    markers = {
        '+': '^',
        '-': 'v'
    }
    
    for _, event in events.iterrows():
        event_date = pd.to_datetime(event['date'])
        if event_date in df.index:
            price = df.loc[event_date, 'Close']
        else:
            nearest_date = df.index[df.index > event_date][0]
            price = df.loc[nearest_date, 'Close']
            event_date = nearest_date
            
        ax1.scatter(event_date, price, 
                   c=colors[event['type']], 
                   marker=markers[event['impact']], 
                   s=150, zorder=5)
        
        ax1.annotate(f"{event['event']}\n{event['detail']}", 
                    xy=(event_date, price),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(facecolor='white', edgecolor=colors[event['type']], alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color=colors[event['type']]))
    
    ax1.set_title('Hundsun Events Analysis', fontsize=14)
    ax1.grid(True)
    
    # 2. 成交量
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(df.index, df['Volume'], color='gray', alpha=0.5)
    ax2.set_title('Volume', fontsize=12)
    ax2.grid(True)
    
    # 3. 事件影响分析
    ax3 = fig.add_subplot(gs[2])
    event_impacts = []
    for _, event in events.iterrows():
        pre_change, post_change = analyze_price_impact(df, event['date'])
        event_impacts.append({
            'event': f"{event['event']}",
            'type': event['type'],
            'pre_change': pre_change,
            'post_change': post_change
        })
    
    impact_df = pd.DataFrame(event_impacts)
    type_impact = impact_df.groupby('type')[['pre_change', 'post_change']].mean()
    x = range(len(type_impact))
    width = 0.35
    
    ax3.bar([i - width/2 for i in x], type_impact['pre_change'], width, 
            label='Pre-10d', color='lightblue')
    ax3.bar([i + width/2 for i in x], type_impact['post_change'], width,
            label='Post-10d', color='lightgreen')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(type_impact.index, rotation=45, ha='right')
    ax3.set_title('Event Impact Analysis (%)', fontsize=12)
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_stock_events()
