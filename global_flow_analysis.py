import yfinance as yf
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import seaborn as sns

class GlobalFlowAnalyzer:
    def __init__(self):
        """初始化全球资金流向分析器"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Heiti TC', 'Microsoft YaHei', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8')
        
        # 自定义样式设置
        mpl.rcParams.update({
            'figure.figsize': [12, 8],
            'figure.dpi': 100,
            'savefig.dpi': 100,
            'font.size': 10,
            'legend.fontsize': 'small',
            'figure.titlesize': 'large',
            'axes.labelsize': 'medium',
            'axes.titlesize': 'medium',
            'xtick.labelsize': 'small',
            'ytick.labelsize': 'small',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'axes.facecolor': '#f0f0f0',
            'figure.facecolor': 'white',
            'axes.edgecolor': '#666666',
            'axes.linewidth': 1.0,
            'lines.linewidth': 1.5,
            'patch.linewidth': 1.0,
            'lines.markersize': 6,
            'lines.markeredgewidth': 1.0,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'xtick.minor.width': 0.5,
            'ytick.minor.width': 0.5,
            'xtick.major.pad': 4,
            'ytick.major.pad': 4
        })
        
        # 使用更现代的颜色方案
        self.colors = [
            '#2196F3',  # 蓝色
            '#FF9800',  # 橙色
            '#4CAF50',  # 绿色
            '#F44336',  # 红色
            '#9C27B0',  # 紫色
            '#00BCD4',  # 青色
            '#FFC107',  # 琥珀色
            '#795548'   # 棕色
        ]
        
        # 标签映射
        self.index_labels = {
            'SPX': '标普500',
            'NDX': '纳斯达克',
            'HSI': '恒生指数',
            'N225': '日经225',
            'FTSE': '富时100',
            'DAX': '德国DAX'
        }
        
        self.commodity_labels = {
            'GOLD': '黄金',
            'OIL': '原油',
            'COPPER': '铜'
        }
        
        self.forex_labels = {
            'USDCNH': '美元/人民币',
            'EURUSD': '欧元/美元',
            'USDJPY': '美元/日元',
            'GBPUSD': '英镑/美元'
        }
        
        self.bond_labels = {
            'US': '美国国债',
            'CN': '中国国债'
        }
        
        # 合并所有标签映射
        self.label_mapping = {}
        for prefix, labels in [
            ('INDEX_', self.index_labels),
            ('COM_', self.commodity_labels),
            ('FX_', self.forex_labels),
            ('BOND_', self.bond_labels)
        ]:
            self.label_mapping.update({f"{prefix}{k}": v for k, v in labels.items()})
        
    def get_global_indices(self):
        """获取全球主要股指数据"""
        indices = {
            'SPX': '^GSPC',    # 标普500
            'NDX': '^IXIC',    # 纳斯达克
            'HSI': '^HSI',     # 恒生指数
            'N225': '^N225',   # 日经225
            'FTSE': '^FTSE',   # 富时100
            'DAX': '^GDAXI'    # 德国DAX
        }
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        dfs = {}
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                df = df.rename(columns={
                    'Close': 'close',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Volume': 'volume'
                })
                dfs[name] = df
                print(f"成功获取 {name} 指数数据")
            except Exception as e:
                print(f"获取 {name} 数据失败: {e}")
        
        return dfs

    def get_commodity_flows(self):
        """获取主要大宗商品数据"""
        try:
            # 获取黄金ETF数据
            gold = yf.Ticker('GLD').history(period='1y')
            print("成功获取黄金ETF数据")
            
            # 获取原油ETF数据
            oil = yf.Ticker('USO').history(period='1y')
            print("成功获取原油ETF数据")
            
            # 获取铜ETF数据
            copper = yf.Ticker('CPER').history(period='1y')
            print("成功获取铜ETF数据")
            
            # 统一列名
            for df in [gold, oil, copper]:
                df.rename(columns={
                    'Close': 'close',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Volume': 'volume'
                }, inplace=True)
            
            return {
                'GOLD': gold,
                'OIL': oil,
                'COPPER': copper
            }
        except Exception as e:
            print(f"获取大宗商品数据失败: {e}")
            return None

    def get_forex_flows(self):
        """获取主要外汇对数据"""
        try:
            # 获取主要货币对数据
            currencies = {
                'USDCNH': 'CNH=X',
                'EURUSD': 'EURUSD=X',
                'USDJPY': 'JPY=X',
                'GBPUSD': 'GBPUSD=X'
            }
            
            forex_data = {}
            for name, symbol in currencies.items():
                df = yf.Ticker(symbol).history(period='1y')
                df.rename(columns={
                    'Close': 'close',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Volume': 'volume'
                }, inplace=True)
                forex_data[name] = df
                print(f"成功获取 {name} 汇率数据")
            
            return forex_data
        except Exception as e:
            print(f"获取外汇数据失败: {e}")
            return None

    def get_bond_yields(self):
        """获取主要国家债券收益率"""
        try:
            # 获取美国10年期国债ETF
            us_bonds = yf.Ticker('IEF').history(period='1y')
            print("成功获取美国国债ETF数据")
            
            # 获取中国国债ETF
            cn_bonds = yf.Ticker('CBON').history(period='1y')
            print("成功获取中国国债ETF数据")
            
            # 统一列名
            for df in [us_bonds, cn_bonds]:
                df.rename(columns={
                    'Close': 'close',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Volume': 'volume'
                }, inplace=True)
            
            return {
                'US': us_bonds,
                'CN': cn_bonds
            }
        except Exception as e:
            print(f"获取债券数据失败: {e}")
            return None

    def calculate_flow_strength(self, data_dict, window=20):
        """计算资金流向强度
        使用价格动量和成交量变化来衡量资金流向强度
        
        Args:
            data_dict: 包含各个市场数据的字典，每个值都是一个DataFrame
            window: 计算动量的窗口期
        """
        flow_strength = {}
        
        for market_type in ['INDEX_', 'COM_', 'FX_', 'BOND_']:
            for name, df in data_dict.items():
                if df is None or df.empty:
                    continue
                    
                key = None
                if market_type == 'INDEX_' and name in self.index_labels:
                    key = f'INDEX_{name}'
                elif market_type == 'COM_' and name in self.commodity_labels:
                    key = f'COM_{name}'
                elif market_type == 'FX_' and name in self.forex_labels:
                    key = f'FX_{name}'
                elif market_type == 'BOND_' and name in self.bond_labels:
                    key = f'BOND_{name}'
                
                if key is not None:
                    try:
                        # 计算价格动量
                        price_momentum = df['close'].pct_change(window)
                        
                        # 计算成交量变化
                        volume_change = df['volume'].pct_change() if 'volume' in df.columns else pd.Series(1, index=df.index)
                        
                        # 计算资金流向强度
                        strength = price_momentum * volume_change
                        
                        # 创建结果DataFrame
                        result_df = pd.DataFrame({
                            'flow_strength': strength
                        })
                        
                        flow_strength[key] = result_df
                    except Exception as e:
                        print(f"计算{key}的资金流向强度时出错: {e}")
        
        return flow_strength

    def analyze_global_flows(self):
        """分析全球资金流向"""
        # 获取各类数据
        indices = self.get_global_indices()
        commodities = self.get_commodity_flows()
        forex = self.get_forex_flows()
        bonds = self.get_bond_yields()
        
        # 合并数据
        data_dict = {**indices, **commodities, **forex, **bonds}
        
        # 计算各市场的资金流向强度
        flow_strength = self.calculate_flow_strength(data_dict)
        
        return flow_strength

    def plot_flow_analysis(self, flow_strength):
        """绘制资金流向分析图"""
        if not flow_strength:
            print("没有足够的数据来绘制分析图")
            return
            
        # 创建子图
        fig = plt.figure(figsize=(16, 22))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.3)
        axes = [fig.add_subplot(gs[i]) for i in range(4)]
        
        # 设置整体标题
        fig.suptitle('全球资金流向分析\n最近30天趋势', 
                    fontsize=20, 
                    y=0.95, 
                    fontproperties='PingFang HK',
                    weight='bold')
        
        # 设置每个子图的数据和样式
        plot_configs = [
            ('INDEX_', 0, '主要股指资金流向强度', '股指'),
            ('COM_', 1, '大宗商品资金流向强度', '商品'),
            ('FX_', 2, '外汇市场资金流向强度', '外汇'),
            ('BOND_', 3, '债券市场资金流向强度', '债券')
        ]
        
        for prefix, idx, title, market in plot_configs:
            ax = axes[idx]
            color_idx = 0
            
            # 绘制数据
            for name, df in flow_strength.items():
                if name.startswith(prefix) and df is not None and not df.empty:
                    color = self.colors[color_idx % len(self.colors)]
                    
                    # 计算移动平均线
                    ma5 = df['flow_strength'].rolling(window=5).mean()
                    ma10 = df['flow_strength'].rolling(window=10).mean()
                    
                    # 绘制原始数据（点状）
                    ax.scatter(df.index[-30:], df['flow_strength'][-30:],
                             color=color, alpha=0.3, s=20,
                             label=f"{self.label_mapping[name]} (日度)")
                    
                    # 绘制移动平均线
                    ax.plot(df.index[-30:], ma5[-30:],
                           color=color, linewidth=2, alpha=0.8,
                           label=f"{self.label_mapping[name]} (5日均线)")
                    ax.plot(df.index[-30:], ma10[-30:],
                           color=color, linewidth=1.5, linestyle='--', alpha=0.6,
                           label=f"{self.label_mapping[name]} (10日均线)")
                    
                    color_idx += 1
            
            # 设置标题和标签
            ax.set_title(title, fontproperties='PingFang HK', pad=20, fontsize=14, weight='bold')
            ax.set_xlabel('日期', fontproperties='PingFang HK', labelpad=10)
            ax.set_ylabel('资金流向强度', fontproperties='PingFang HK', labelpad=10)
            
            # 设置网格
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # 设置图例
            ax.legend(prop={'family': 'PingFang HK', 'size': 9}, 
                     loc='center left', 
                     bbox_to_anchor=(1.02, 0.5),
                     borderaxespad=0,
                     frameon=True,
                     fancybox=True,
                     shadow=True)
            
            # 设置x轴日期格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 添加零线
            ax.axhline(y=0, color='#666666', linestyle='-', alpha=0.3, zorder=1)
            
            # 设置y轴范围，使其对称
            y_values = []
            for df in flow_strength.values():
                if df is not None and not df.empty:
                    values = df['flow_strength'].dropna()
                    if not values.empty:
                        y_values.extend(abs(values))
            
            if y_values:
                max_abs_y = max(y_values)
                if max_abs_y > 0 and not np.isinf(max_abs_y) and not np.isnan(max_abs_y):
                    ax.set_ylim(-max_abs_y * 1.1, max_abs_y * 1.1)
            
            # 添加市场类型标签
            ax.text(0.02, 0.95, market, 
                   transform=ax.transAxes,
                   fontproperties='PingFang HK',
                   fontsize=12,
                   weight='bold',
                   bbox=dict(facecolor='white', 
                            edgecolor='none',
                            alpha=0.8,
                            pad=3))
        
        # 添加说明文字
        fig.text(0.02, 0.02, 
                '说明：\n'
                '• 资金流向强度 = 价格动量 × 成交量变化\n'
                '• 正值（向上）表示资金流入，负值（向下）表示资金流出\n'
                '• 实线为5日均线，虚线为10日均线，散点为日度数据\n'
                '• 数据每日更新，展示最近30天走势',
                fontproperties='PingFang HK', 
                fontsize=10,
                bbox=dict(facecolor='white', 
                         edgecolor='#666666',
                         alpha=0.8,
                         pad=10,
                         boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.show()

    def analyze_correlation(self, flow_strength):
        """分析各市场之间的相关性"""
        # 提取最近30天的flow_strength数据
        market_flows = {}
        for name, df in flow_strength.items():
            if df is not None and not df.empty:
                market_flows[self.label_mapping[name]] = df['flow_strength'][-30:]
        
        # 创建相关性矩阵
        flow_df = pd.DataFrame(market_flows)
        corr_matrix = flow_df.corr()
        
        # 创建掩码，只显示上三角矩阵
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # 设置图形大小和样式
        plt.figure(figsize=(16, 12))
        
        # 创建自定义颜色映射
        colors = ['#4575B4', '#91BFDB', '#E0F3F8', '#FFFFBF', '#FEE090', '#FC8D59', '#D73027']
        n_colors = len(colors)
        custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom", colors)
        
        # 绘制热力图
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap=custom_cmap,
                   center=0,
                   vmin=-1,
                   vmax=1,
                   fmt='.2f',
                   square=True,
                   linewidths=0.5,
                   cbar_kws={
                       'label': '相关系数',
                       'orientation': 'horizontal',
                       'pad': 0.2,
                       'aspect': 30,
                       'shrink': 0.8,
                       'ticks': np.linspace(-1, 1, n_colors)
                   },
                   annot_kws={'size': 8})
        
        plt.title('全球市场资金流向相关性分析', 
                 fontproperties='PingFang HK', 
                 pad=20,
                 fontsize=16,
                 weight='bold')
        
        # 调整标签字体
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 添加说明文字
        plt.figtext(0.02, 0.02,
                   '说明：\n'
                   '• 相关系数范围：-1到1\n'
                   '• 红色表示正相关，蓝色表示负相关\n'
                   '• 颜色越深表示相关性越强\n'
                   '• 基于最近30天的数据计算\n'
                   '• 上三角矩阵显示，避免重复信息',
                   fontproperties='PingFang HK',
                   fontsize=10,
                   bbox=dict(facecolor='white', 
                            edgecolor='#666666',
                            alpha=0.8,
                            pad=10,
                            boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analyzer = GlobalFlowAnalyzer()
    flow_strength = analyzer.analyze_global_flows()
    analyzer.plot_flow_analysis(flow_strength)
    analyzer.analyze_correlation(flow_strength)
