import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

class PalmOilAnalyzer:
    def __init__(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 对于macOS
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置图表样式
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
    def create_sample_data(self):
        """创建示例数据"""
        # 生成最近36个月的日期
        dates = pd.date_range(end=datetime.now(), periods=36, freq='M')
        
        # 创建示例库存数据（单位：万吨）
        base_inventory = 300  # 基础库存水平
        
        # 添加季节性波动（每年一个完整周期）
        t = np.linspace(0, 6*np.pi, 36)  # 三年的周期
        seasonal_pattern = 30 * np.sin(t)  # 季节性波动
        
        # 添加长期趋势（略微上升）
        trend = np.linspace(0, 45, 36)  # 三年内总体上涨45万吨
        
        # 添加随机波动
        random_fluctuation = np.random.normal(0, 15, 36)  # 增加随机波动幅度
        
        # 添加一些特殊事件的影响（如突发事件导致的库存变化）
        special_events = np.zeros(36)
        special_events[15] = 50  # 第16个月突然增加
        special_events[25] = -30  # 第26个月突然减少
        
        inventory = base_inventory + seasonal_pattern + trend + random_fluctuation + special_events
        
        # 确保库存始终为正
        inventory = np.maximum(inventory, 50)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'inventory': inventory,
            'mom_change': [0] + [inventory[i] - inventory[i-1] for i in range(1, len(inventory))]
        })
        
        return df
    
    def format_ytick(self, value, pos):
        """格式化y轴标签"""
        return f'{int(value)}万吨'
    
    def plot_inventory(self, df):
        """绘制库存图表"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
        fig.suptitle('棕榈油库存分析（三年数据）', fontsize=16, y=0.95)
        
        # 设置日期格式
        date_formatter = mdates.DateFormatter('%Y-%m')
        
        # 绘制库存量（上图）
        color = '#1f77b4'  # 主曲线颜色
        ax1.plot(df['date'], df['inventory'], color=color, linewidth=2.5, label='库存量')
        ax1.fill_between(df['date'], df['inventory'], alpha=0.2, color=color)
        
        # 计算并绘制移动平均线
        df['MA3'] = df['inventory'].rolling(window=3).mean()
        df['MA6'] = df['inventory'].rolling(window=6).mean()
        df['MA12'] = df['inventory'].rolling(window=12).mean()
        
        ax1.plot(df['date'], df['MA3'], '#ff7f0e', linestyle='--', label='3月移动平均')
        ax1.plot(df['date'], df['MA6'], '#2ca02c', linestyle='--', label='6月移动平均')
        ax1.plot(df['date'], df['MA12'], '#d62728', linestyle='--', label='12月移动平均')
        
        # 添加年度最高最低点标注
        for year in df['date'].dt.year.unique():
            year_data = df[df['date'].dt.year == year]
            max_idx = year_data['inventory'].idxmax()
            min_idx = year_data['inventory'].idxmin()
            
            # 标注最高点
            ax1.annotate(f'{year_data["inventory"].max():.0f}',
                        xy=(df['date'][max_idx], df['inventory'][max_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # 标注最低点
            ax1.annotate(f'{year_data["inventory"].min():.0f}',
                        xy=(df['date'][min_idx], df['inventory'][min_idx]),
                        xytext=(10, -10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 设置上图格式
        ax1.set_title('月度库存量', pad=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        ax1.xaxis.set_major_formatter(date_formatter)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每3个月显示一个刻度
        ax1.yaxis.set_major_formatter(FuncFormatter(self.format_ytick))
        
        # 绘制环比变化（下图）
        bars = ax2.bar(df['date'], df['mom_change'],
                      color=['#d62728' if x < 0 else '#2ca02c' for x in df['mom_change']])
        
        # 添加零线
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.,
                    height if height >= 0 else height - 2,
                    f'{height:.0f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)
        
        # 设置下图格式
        ax2.set_title('月度环比变化', pad=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.xaxis.set_major_formatter(date_formatter)
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax2.yaxis.set_major_formatter(FuncFormatter(self.format_ytick))
        
        # 调整布局
        plt.tight_layout()
        return fig

    def analyze_inventory(self):
        """分析库存数据"""
        # 获取示例数据
        df = self.create_sample_data()
        
        # 绘制图表
        fig = self.plot_inventory(df)
        
        # 显示图表
        plt.show()
        
        # 基本统计分析
        latest_inventory = df['inventory'].iloc[-1]
        avg_inventory = df['inventory'].mean()
        max_inventory = df['inventory'].max()
        min_inventory = df['inventory'].min()
        
        print("\n棕榈油库存统计分析")
        print("-" * 30)
        print(f"当前库存：{latest_inventory:.1f}万吨")
        print(f"平均库存：{avg_inventory:.1f}万吨")
        print(f"最高库存：{max_inventory:.1f}万吨")
        print(f"最低库存：{min_inventory:.1f}万吨")
        print(f"库存波动范围：{max_inventory-min_inventory:.1f}万吨")

if __name__ == "__main__":
    analyzer = PalmOilAnalyzer()
    analyzer.analyze_inventory()
