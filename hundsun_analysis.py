import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置中文字体为Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class StockAnalyzer:
    def __init__(self, stock_code):
        """
        初始化分析器
        :param stock_code: 股票代码，例如：'600570'
        """
        self.stock_code = stock_code
        
    def get_data(self):
        """
        获取股票数据
        :return: pandas DataFrame 包含股票历史数据
        :raises: Exception 当数据获取失败时
        """
        try:
            # 获取当前日期
            end_date = datetime.now()
            # 获取3年前的日期
            start_date = end_date - timedelta(days=3*365)
            
            # 格式化日期
            start_date_str = start_date.strftime('%Y%m%d')
            end_date_str = end_date.strftime('%Y%m%d')
            
            # 获取日K数据
            df = ak.stock_zh_a_hist(symbol=self.stock_code, 
                                  start_date=start_date_str,
                                  end_date=end_date_str,
                                  adjust="qfq")
            
            # 检查数据是否为空
            if df.empty:
                raise Exception(f"未能获取到股票 {self.stock_code} 的数据")
                
            # 重命名列以匹配之前的代码
            df = df.rename(columns={
                '日期': 'trade_date',
                '收盘': 'close',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'vol',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_chg',
                '涨跌额': 'change'
            })
            
            # 确保日期列是datetime类型
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # 按日期排序
            df = df.sort_values('trade_date')
            
            # 重置索引
            df = df.reset_index(drop=True)
            
            print(f"成功获取{len(df)}条交易数据，日期范围：{df['trade_date'].min()} 至 {df['trade_date'].max()}")
            
            return df
            
        except Exception as e:
            raise Exception(f"获取股票数据失败: {str(e)}")

    def check_divergence(self, df):
        """检测量价背离"""
        df['price_change'] = df['close'].diff()
        df['volume_change'] = df['vol'].diff()
        
        # 计算价格和成交量的变化率
        df['price_change_rate'] = df['price_change'] / df['close'].shift(1) * 100
        df['volume_change_rate'] = df['volume_change'] / df['vol'].shift(1) * 100
        
        # 判断背离（价格下跌但成交量增加）
        divergence = (df['price_change'] < 0) & (df['volume_change'] > 0)
        
        # 计算背离强度
        df['divergence_strength'] = 0.0
        mask = divergence
        df.loc[mask, 'divergence_strength'] = (
            abs(df.loc[mask, 'price_change_rate']) * 
            df.loc[mask, 'volume_change_rate'] / 100
        )
        
        return df[divergence]

    def calc_money_flow(self, df):
        """
        计算资金流向
        采用价格位置加权的方法计算主动买入比例
        考虑因素：
        1. 收盘价在当日价格区间的位置
        2. 成交量的变化
        3. 开盘价和收盘价的关系
        """
        # 计算基础指标
        df['price_range'] = df['high'] - df['low']  # 价格区间
        df['vol_ma5'] = df['vol'].rolling(5).mean()  # 5日平均成交量
        
        def calculate_buy_ratio(row):
            # 如果最高价等于最低价，返回0.5
            if row['price_range'] == 0:
                return 0.5
                
            # 计算收盘价在当日价格区间的位置 (0-1)
            close_position = (row['close'] - row['low']) / row['price_range']
            
            # 计算开盘价在当日价格区间的位置 (0-1)
            open_position = (row['open'] - row['low']) / row['price_range']
            
            # 基础主动买入比例
            if row['close'] >= row['open']:
                # 收盘价大于开盘价，上涨
                base_ratio = 0.5 + (close_position - open_position) * 0.5
            else:
                # 收盘价小于开盘价，下跌
                base_ratio = 0.5 - (open_position - close_position) * 0.5
            
            # 成交量因子：当成交量大于5日平均时，放大买入信号
            vol_factor = 1.0
            if not pd.isna(row['vol_ma5']) and row['vol_ma5'] > 0:
                vol_ratio = row['vol'] / row['vol_ma5']
                if vol_ratio > 1:
                    # 成交量放大时，根据涨跌调整买入信号
                    if row['close'] > row['open']:
                        vol_factor = min(1.2, 1 + (vol_ratio - 1) * 0.1)
                    else:
                        vol_factor = max(0.8, 1 - (vol_ratio - 1) * 0.1)
            
            # 最终主动买入比例
            final_ratio = base_ratio * vol_factor
            
            # 确保结果在0-1之间
            return max(0, min(1, final_ratio))
        
        # 计算每日主动买入比例
        df['buy_ratio'] = df.apply(calculate_buy_ratio, axis=1)
        
        # 计算移动平均
        df['buy_ratio_ma5'] = df['buy_ratio'].rolling(window=5, center=False).mean()
        
        # 添加趋势指标
        df['buy_trend'] = df['buy_ratio_ma5'].diff()
        
        return df

    def calculate_technical_indicators(self, df):
        """计算技术指标"""
        df_tech = df.copy()
        
        # 计算移动平均线
        windows = [5, 10, 20, 60]
        for window in windows:
            df_tech[f'MA{window}'] = df_tech['close'].rolling(window=window).mean()
        
        # 计算价格变化
        df_tech['Price_Change'] = df_tech['close'].pct_change()
        df_tech['Price_Change_5'] = df_tech['close'].pct_change(5)
        df_tech['Price_Change_10'] = df_tech['close'].pct_change(10)
        
        # 计算RSI
        for window in [6, 12, 24]:
            delta = df_tech['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df_tech[f'RSI_{window}'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        for window in [20, 40]:
            df_tech[f'BB_Middle_{window}'] = df_tech['close'].rolling(window=window).mean()
            df_tech[f'BB_Upper_{window}'] = df_tech[f'BB_Middle_{window}'] + 2 * df_tech['close'].rolling(window=window).std()
            df_tech[f'BB_Lower_{window}'] = df_tech[f'BB_Middle_{window}'] - 2 * df_tech['close'].rolling(window=window).std()
            df_tech[f'BB_Width_{window}'] = (df_tech[f'BB_Upper_{window}'] - df_tech[f'BB_Lower_{window}']) / df_tech[f'BB_Middle_{window}']
            df_tech[f'BB_Position_{window}'] = (df_tech['close'] - df_tech[f'BB_Lower_{window}']) / (df_tech[f'BB_Upper_{window}'] - df_tech[f'BB_Lower_{window}'])
        
        # 计算MACD
        exp1 = df_tech['close'].ewm(span=12, adjust=False).mean()
        exp2 = df_tech['close'].ewm(span=26, adjust=False).mean()
        df_tech['MACD'] = exp1 - exp2
        df_tech['Signal_Line'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
        df_tech['MACD_Hist'] = df_tech['MACD'] - df_tech['Signal_Line']
        df_tech['MACD_Norm'] = df_tech['MACD'] / df_tech['close']
        
        # 计算成交量指标
        df_tech['VOL_MA20'] = df_tech['vol'].rolling(window=20).mean()
        df_tech['VOL_Ratio'] = df_tech['vol'] / df_tech['VOL_MA20']
        df_tech['VOL_Change'] = df_tech['vol'].pct_change()
        df_tech['VOL_MA_Ratio'] = df_tech['vol'] / df_tech['vol'].rolling(window=20).mean()
        
        # 计算KDJ
        low_9 = df_tech['low'].rolling(window=9).min()
        high_9 = df_tech['high'].rolling(window=9).max()
        df_tech['RSV'] = (df_tech['close'] - low_9) / (high_9 - low_9) * 100
        df_tech['K'] = df_tech['RSV'].ewm(com=2).mean()
        df_tech['D'] = df_tech['K'].ewm(com=2).mean()
        df_tech['J'] = 3 * df_tech['K'] - 2 * df_tech['D']
        
        # 计算ADX
        tr1 = df_tech['high'] - df_tech['low']
        tr2 = abs(df_tech['high'] - df_tech['close'].shift(1))
        tr3 = abs(df_tech['low'] - df_tech['close'].shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        dx = abs(df_tech['high'].diff(1) - df_tech['low'].diff(1)) / tr
        df_tech['ADX'] = dx.rolling(window=14).mean() * 100
        df_tech['ADX_Norm'] = df_tech['ADX'] / 100
        
        # 计算ATR
        df_tech['ATR'] = tr.rolling(window=14).mean()
        df_tech['ATR_Ratio'] = df_tech['ATR'] / df_tech['close']
        
        # 计算价格位置指标
        highest = df_tech['high'].rolling(window=20).max()
        lowest = df_tech['low'].rolling(window=20).min()
        df_tech['Price_Position'] = (df_tech['close'] - lowest) / (highest - lowest)
        
        # 计算价格和均线距离
        df_tech['Price_MA_Distance'] = (df_tech['close'] - df_tech['MA20']) / df_tech['MA20']
        
        # 计算主动买入比例（基于日内最高最低价的位置）
        df_tech['buy_ratio'] = (df_tech['close'] - df_tech['low']) / (df_tech['high'] - df_tech['low']) * 100
        df_tech['buy_ratio_ma5'] = df_tech['buy_ratio'].rolling(window=5).mean()
        df_tech['buy_pressure'] = df_tech['buy_ratio'] / df_tech['buy_ratio_ma5']
        
        return df_tech

    def create_features(self, df):
        """创建预测特征"""
        features = [
            # 移动平均线
            'MA5', 'MA10', 'MA20', 'MA60',
            
            # 价格变化
            'Price_Change', 'Price_Change_5', 'Price_Change_10',
            
            # RSI指标
            'RSI_6', 'RSI_12', 'RSI_24',
            
            # 布林带指标
            'BB_Position_20', 'BB_Position_40',
            'BB_Width_20', 'BB_Width_40',
            
            # MACD指标
            'MACD_Norm', 'MACD_Hist',
            
            # 成交量指标
            'VOL_Ratio', 'VOL_Change', 'VOL_MA_Ratio',
            
            # KDJ指标
            'K', 'D', 'J',
            
            # 趋势强度
            'ADX_Norm',
            
            # 波动性
            'ATR_Ratio',
            
            # 市场情绪
            'Price_Position', 'Price_MA_Distance',
            
            # 主动买入
            'buy_ratio', 'buy_ratio_ma5', 'buy_pressure'
        ]
        
        return features

    def _update_prediction_features(self, features, pred_price, prev_price=None, step=1):
        """智能更新预测特征"""
        features = features.copy()
        
        # 计算价格变化
        if prev_price is not None:
            price_change = (pred_price - prev_price) / prev_price
        else:
            price_change = 0
            
        # 更新移动平均线
        alpha = 2.0 / (step + 1)  # 指数加权因子
        for ma in ['MA5', 'MA10', 'MA20', 'MA60']:
            features[ma] = float(features[ma]) * (1 - alpha) + pred_price * alpha
        
        # 更新价格变化指标
        features['Price_Change'] = price_change
        features['Price_Change_5'] = float(features['Price_Change_5']) * 0.8 + price_change * 0.2
        features['Price_Change_10'] = float(features['Price_Change_10']) * 0.9 + price_change * 0.1
        
        # 更新RSI
        if price_change > 0:
            gain = price_change
            loss = 0
        else:
            gain = 0
            loss = -price_change
            
        for window in [6, 12, 24]:
            old_rsi = float(features[f'RSI_{window}'])
            new_rsi = old_rsi * (window-1)/window + (100 * gain)/(gain + loss + 1e-9) / window
            features[f'RSI_{window}'] = min(max(new_rsi, 30), 70)  # 限制在合理范围内
        
        # 更新布林带位置
        for window in [20, 40]:
            pos = float(features[f'BB_Position_{window}'])
            features[f'BB_Position_{window}'] = min(max(
                pos * 0.7 + 0.5 * 0.3,  # 向中线回归
                0.2), 0.8)  # 限制在合理范围内
            features[f'BB_Width_{window}'] = float(features[f'BB_Width_{window}']) * 0.95  # 稳定收窄
        
        # 更新MACD相关指标
        features['MACD_Norm'] = float(features['MACD_Norm']) * 0.9  # 逐渐衰减
        features['MACD_Hist'] = float(features['MACD_Hist']) * 0.9
        
        # 更新成交量指标（假设成交量逐渐回归均值）
        features['VOL_Ratio'] = float(features['VOL_Ratio']) * 0.8 + 1.0 * 0.2
        features['VOL_Change'] = float(features['VOL_Change']) * 0.8
        features['VOL_MA_Ratio'] = float(features['VOL_MA_Ratio']) * 0.8 + 1.0 * 0.2
        
        # 更新KDJ（向中性位置收敛）
        for indicator in ['K', 'D', 'J']:
            features[indicator] = float(features[indicator]) * 0.8 + 50 * 0.2
        
        # 更新趋势强度（假设趋势强度逐渐减弱）
        features['ADX_Norm'] = float(features['ADX_Norm']) * 0.95
        
        # 更新波动性（假设波动性逐渐降低）
        features['ATR_Ratio'] = float(features['ATR_Ratio']) * 0.9 + 1.0 * 0.1
        
        # 更新市场情绪（向中性位置收敛）
        features['Price_Position'] = float(features['Price_Position']) * 0.8 + 0.5 * 0.2
        features['Price_MA_Distance'] = float(features['Price_MA_Distance']) * 0.7
        
        # 更新主动买入指标（向中性位置收敛）
        features['buy_ratio'] = float(features['buy_ratio']) * 0.8 + 50 * 0.2
        features['buy_ratio_ma5'] = float(features['buy_ratio_ma5']) * 0.8 + 50 * 0.2
        features['buy_pressure'] = float(features['buy_pressure']) * 0.7
        
        return features

    def predict_price(self, df, future_days=30):
        """
        预测未来价格
        使用集成学习方法和多个技术指标进行预测
        """
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import RobustScaler
        from sklearn.model_selection import TimeSeriesSplit
        import warnings
        warnings.filterwarnings('ignore')

        # 准备数据
        df_pred = self.calculate_technical_indicators(df)
        features = self.create_features(df_pred)
        
        # 删除包含NaN的行
        df_pred = df_pred.dropna()
        
        # 准备特征和目标变量
        X = df_pred[features]
        y = df_pred['close']
        
        # 使用稳健缩放器处理异常值
        scaler_X = RobustScaler()
        scaler_y = RobustScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
        
        # 创建多个模型
        models = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
            'et': ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42),
            'lasso': LassoCV(cv=5, random_state=42)
        }
        
        # 使用时间序列交叉验证训练模型
        tscv = TimeSeriesSplit(n_splits=5)
        model_scores = {}
        model_predictions = {}
        
        # 训练模型
        for name, model in models.items():
            scores = []
            predictions = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
                
                model.fit(X_train, y_train.ravel())
                score = model.score(X_val, y_val)
                pred = model.predict(X_scaled)
                
                scores.append(score)
                predictions.append(pred)
            
            model_scores[name] = np.mean(scores)
            model_predictions[name] = np.mean(predictions, axis=0)
        
        # 根据验证分数为模型分配权重
        total_score = sum([max(0.1, score) for score in model_scores.values()])
        weights = {name: max(0.1, score)/total_score for name, score in model_scores.items()}
        
        # 准备未来预测
        last_data = df_pred.iloc[-1:][features]
        future_predictions = []
        future_dates = pd.date_range(start=df_pred.index[-1], periods=future_days+1, freq='D')[1:]
        prediction_stds = []
        
        # 获取最近的趋势
        recent_trend = df_pred['close'].pct_change().rolling(window=5).mean().iloc[-1]
        trend_factor = np.clip(recent_trend * 20, -0.02, 0.02)  # 限制趋势影响
        
        # 获取最近的价格
        last_price = df_pred['close'].iloc[-1]
        current_price = last_price
        
        # 逐日预测
        current_features = last_data.copy()
        for step in range(future_days):
            # 标准化当前特征
            current_scaled = scaler_X.transform(current_features)
            
            # 集成预测
            model_preds = []
            pred_scaled = np.zeros(1)
            for name, model in models.items():
                pred = model.predict(current_scaled)
                model_preds.append(pred[0])
                pred_scaled += pred * weights[name]
            
            # 计算预测标准差
            pred_std = np.std(model_preds)
            prediction_stds.append(pred_std)
            
            # 转换回原始比例并应用趋势调整
            base_pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            trend_adjustment = base_pred * trend_factor * (1 - step/future_days)  # 趋势影响随时间衰减
            pred_price = base_pred + trend_adjustment
            
            # 限制单日价格变化
            max_daily_change = 0.1  # 最大允许10%的日变化
            pred_price = current_price * (1 + np.clip(
                (pred_price - current_price) / current_price,
                -max_daily_change,
                max_daily_change
            ))
            
            future_predictions.append(pred_price)
            
            # 更新特征用于下一次预测
            current_features = self._update_prediction_features(
                current_features, pred_price, current_price, step+1)
            current_price = pred_price
        
        # 创建预测结果DataFrame
        future_df = pd.DataFrame({
            'predicted_price': future_predictions,
            'prediction_std': prediction_stds,
            'date': future_dates
        })
        
        # 计算综合置信度
        confidence = np.mean(list(model_scores.values()))
        
        # 计算预测区间
        avg_std = np.mean(prediction_stds)
        
        return future_df, confidence, avg_std, weights

    def plot_prediction(self, df, future_df, confidence, weights):
        """绘制预测结果"""
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
        
        # 计算均线
        df_plot = df.copy()
        df_plot['MA20'] = df_plot['close'].rolling(window=20).mean()
        df_plot['MA60'] = df_plot['close'].rolling(window=60).mean()
        
        # 主图：价格和预测
        ax1.plot(df_plot.index, df_plot['close'], label='历史价格', color='blue')
        ax1.plot(future_df['date'], future_df['predicted_price'], 
                label=f'预测价格 (置信度: {confidence:.2%})', 
                color='red', linestyle='--')
        
        # 添加预测区间
        ax1.fill_between(future_df['date'],
                        future_df['predicted_price'] - 2*future_df['prediction_std'],
                        future_df['predicted_price'] + 2*future_df['prediction_std'],
                        color='red', alpha=0.1,
                        label='95%预测区间')
        
        # 添加关键均线
        ax1.plot(df_plot.index, df_plot['MA20'], label='20日均线', color='green', alpha=0.5)
        ax1.plot(df_plot.index, df_plot['MA60'], label='60日均线', color='purple', alpha=0.5)
        
        ax1.set_title('恒生电子价格预测')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('价格')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # 子图：模型权重
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        ax2.bar(weights.keys(), weights.values(), color=colors)
        ax2.set_title('模型权重分布')
        ax2.set_ylabel('权重')
        
        plt.tight_layout()
        plt.show()

    def plot_analysis(self, df):
        """可视化分析结果"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建四个子图
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16), height_ratios=[3, 2, 2, 2])
        
        # 1. 绘制K线和背离点
        ax1.plot(df.index, df['close'], label='收盘价', color='#1f77b4')
        
        # 标注背离点
        divergence_points = df[df['divergence_strength'] > 0]
        if not divergence_points.empty:
            # 绘制背离点
            ax1.scatter(divergence_points.index, divergence_points['close'], 
                       color='red', marker='^', s=100, label='量价背离')
            
            # 为重要背离点添加标注
            for idx, row in divergence_points.iterrows():
                if row['divergence_strength'] > divergence_points['divergence_strength'].mean():
                    ax1.annotate(f'背离强度: {row["divergence_strength"]:.1f}',
                                xy=(idx, row['close']),
                                xytext=(10, 10), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax1.set_title('恒生电子 (600570) 分析')
        ax1.set_ylabel('价格')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # 2. 绘制成交量
        colors = ['red' if c >= o else 'green' for c, o in zip(df['close'], df['open'])]
        ax2.bar(df.index, df['vol'], color=colors, label='成交量', alpha=0.7)
        ax2.plot(df.index, df['vol_ma5'], color='blue', label='5日均量', linestyle='--')
        
        # 标注异常成交量
        vol_mean = df['vol'].mean()
        vol_std = df['vol'].std()
        unusual_vol = df[df['vol'] > vol_mean + 2*vol_std]
        for idx, row in unusual_vol.iterrows():
            ax2.annotate(f'放量{row["vol"]/vol_mean:.1f}倍',
                        xy=(idx, row['vol']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        
        ax2.set_ylabel('成交量')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

        # 3. 绘制主动买入比例
        ax3.plot(df.index, df['buy_ratio'], label='主动买入比例', color='gray', alpha=0.5)
        ax3.plot(df.index, df['buy_ratio_ma5'], label='5日主动买入均线', color='#d62728', linewidth=2)
        ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
        ax3.fill_between(df.index, 0.5, df['buy_ratio_ma5'], 
                        where=(df['buy_ratio_ma5'] >= 0.5),
                        color='red', alpha=0.1)
        ax3.fill_between(df.index, 0.5, df['buy_ratio_ma5'],
                        where=(df['buy_ratio_ma5'] < 0.5),
                        color='green', alpha=0.1)
        ax3.set_ylabel('主动买入比例')
        ax3.set_ylim(0, 1)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()

        # 4. 绘制背离强度
        if 'divergence_strength' in df.columns:
            bars = ax4.bar(df.index, df['divergence_strength'], 
                          color='purple', alpha=0.5, label='背离强度')
            ax4.set_ylabel('背离强度')
            # 添加背离强度标签
            for idx, row in df[df['divergence_strength'] > 0].iterrows():
                ax4.text(idx, row['divergence_strength'], 
                        f'{row["divergence_strength"]:.1f}',
                        ha='center', va='bottom')
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.legend()

        plt.tight_layout()
        plt.show()
        
        # 打印最近的主动买入情况
        recent_days = 5
        print(f"\n最近{recent_days}天主动买入情况:")
        recent_data = df.tail(recent_days)
        for date, row in recent_data.iterrows():
            trend = "上升" if row['buy_trend'] > 0 else "下降" if row['buy_trend'] < 0 else "持平"
            print(f"日期: {row['trade_date']}, 收盘: {row['close']:.2f}, "
                  f"主动买入: {row['buy_ratio']:.2f}%, 5日均值: {row['buy_ratio_ma5']:.2f}%, "
                  f"趋势: {trend}")
        
        # 打印显著的量价背离点
        significant_divergence = df[df['divergence_strength'] > 0].sort_values('divergence_strength', ascending=False)
        if not significant_divergence.empty:
            print("\n显著的量价背离点:")
            for _, row in significant_divergence.head().iterrows():
                print(f"日期: {row['trade_date']}, "
                      f"收盘价: {row['close']:.2f}, "
                      f"背离强度: {row['divergence_strength']:.1f}, "
                      f"成交量变化: {row['volume_change_rate']:.1f}%, "
                      f"价格变化: {row['price_change_rate']:.1f}%")

    def plot_wyckoff_analysis(self, df):
        """绘制威科夫分析图"""
        # 创建子图
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), 
                                          gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle('威科夫分析', fontsize=16)
        
        # 绘制K线图
        dates = pd.to_datetime(df['trade_date'])
        
        # 计算移动平均线
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        # 绘制收盘价和均线
        ax1.plot(dates, df['close'], label='收盘价', linewidth=1)
        ax1.plot(dates, df['MA5'], label='MA5', linewidth=1)
        ax1.plot(dates, df['MA10'], label='MA10', linewidth=1)
        ax1.plot(dates, df['MA20'], label='MA20', linewidth=1)
        
        # 获取威科夫分析结果
        analysis = self.analyze_wyckoff(df)
        
        # 添加支撑位和阻力位
        ax1.axhline(y=analysis['support'], color='g', linestyle='--', label='支撑位')
        ax1.axhline(y=analysis['resistance'], color='r', linestyle='--', label='阻力位')
        
        # 标注当前阶段
        ax1.text(0.02, 0.98, f"当前阶段: {analysis['phase']}\n"
                 f"趋势强度: {analysis['trend_strength']}/4\n"
                 f"价格位置: {analysis['price_position']:.1%}",
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_title('价格走势')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # 绘制成交量
        ax2.bar(dates, df['vol'], label='成交量', alpha=0.7)
        ax2.set_title('成交量')
        ax2.grid(True)
        
        # 绘制主动买入比例
        df_tech = self.calculate_technical_indicators(df)
        ax3.plot(dates, df_tech['buy_ratio'], label='主动买入比例', color='orange')
        ax3.plot(dates, df_tech['buy_ratio_ma5'], label='5日均值', color='blue')
        ax3.set_title('主动买入比例')
        ax3.set_ylim(0, 100)
        ax3.grid(True)
        ax3.legend()
        
        # 调整布局
        plt.tight_layout()
        plt.show()

    def analyze_wyckoff(self, df):
        """
        威科夫分析
        返回：阶段、趋势强度、支撑位、阻力位
        """
        df = df.copy()
        
        # 计算价格波动
        df['price_change'] = df['close'].pct_change()
        df['price_std'] = df['close'].rolling(window=20).std()
        
        # 计算成交量指标
        df['volume_ma20'] = df['vol'].rolling(window=20).mean()
        df['volume_std'] = df['vol'].rolling(window=20).std()
        df['volume_ratio'] = df['vol'] / df['volume_ma20']
        
        # 计算支撑和阻力位
        window = 20
        df['support'] = df['close'].rolling(window=window).min()
        df['resistance'] = df['close'].rolling(window=window).max()
        
        # 识别趋势
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        
        # 计算趋势强度
        trend_strength = 0
        recent_data = df.tail(20)
        
        # 基于均线判断趋势
        if recent_data['ma20'].iloc[-1] > recent_data['ma50'].iloc[-1]:
            trend_strength += 1
        if recent_data['ma20'].iloc[-1] > recent_data['ma20'].iloc[-5]:
            trend_strength += 1
            
        # 基于成交量判断趋势
        if recent_data['volume_ratio'].mean() > 1:
            trend_strength += 1
        
        # 基于价格位置判断趋势
        price_position = (recent_data['close'].iloc[-1] - recent_data['support'].iloc[-1]) / \
                        (recent_data['resistance'].iloc[-1] - recent_data['support'].iloc[-1])
        if price_position > 0.5:
            trend_strength += 1
            
        # 识别威科夫阶段
        recent_vol = recent_data['vol'].mean()
        recent_price_range = recent_data['close'].max() - recent_data['close'].min()
        recent_price_trend = recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]
        
        if recent_vol < df['vol'].mean() * 0.8 and abs(recent_price_trend) < recent_data['price_std'].mean():
            phase = "积累期"
        elif recent_vol > df['vol'].mean() * 1.2 and recent_price_trend > 0:
            phase = "上升期"
        elif recent_vol > df['vol'].mean() * 1.2 and recent_price_trend < 0:
            phase = "分配期"
        elif recent_vol < df['vol'].mean() * 0.8 and recent_price_trend < 0:
            phase = "下降期"
        else:
            phase = "整理期"
            
        # 获取关键价位
        support_level = recent_data['support'].iloc[-1]
        resistance_level = recent_data['resistance'].iloc[-1]
        
        return {
            'phase': phase,
            'trend_strength': trend_strength,
            'support': support_level,
            'resistance': resistance_level,
            'price_position': price_position
        }

    def predict_wyckoff_price(self, df, future_days=30):
        """根据威科夫理论预测价格"""
        wyckoff_analysis = self.analyze_wyckoff(df)
        
        # 获取最新日期
        last_date = pd.to_datetime(df['trade_date'].iloc[-1])
        
        # 生成未来日期
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                   periods=future_days, freq='D')
        
        # 创建预测数据框
        future_df = pd.DataFrame({
            'date': future_dates,
            'predicted_price': None,
            'wyckoff_phase': wyckoff_analysis['phase']
        })
        
        # 获取最新价格
        last_price = df['close'].iloc[-1]
        
        # 根据当前阶段调整预测
        phase = wyckoff_analysis['phase']
        trend_strength = wyckoff_analysis['trend_strength']
        support = wyckoff_analysis['support']
        resistance = wyckoff_analysis['resistance']
        
        # 计算每日价格变化
        if phase == '上升期':
            daily_change = (resistance - last_price) / future_days * trend_strength/4
        elif phase == '下降期':
            daily_change = (support - last_price) / future_days * trend_strength/4
        else:
            daily_change = 0
            
        # 生成预测价格
        prices = []
        current_price = last_price
        for _ in range(future_days):
            current_price += daily_change
            if phase == '上升期':
                current_price = min(current_price, resistance)
            elif phase == '下降期':
                current_price = max(current_price, support)
            prices.append(current_price)
            
        future_df['predicted_price'] = prices
        
        return future_df

    def predict_price(self, df, future_days=30):
        """
        预测未来价格
        结合机器学习模型和威科夫分析
        """
        # 获取机器学习模型预测
        ml_future_df, confidence, _, weights = self._predict_price_ml(df, future_days)
        
        # 获取威科夫预测
        wyckoff_predictions = self.predict_wyckoff_price(df, future_days)
        
        # 综合两种预测
        combined_predictions = []
        for i in range(future_days):
            ml_pred = ml_future_df['predicted_price'].iloc[i]
            wyckoff_pred = wyckoff_predictions['predicted_price'].iloc[i]
            
            # 根据置信度调整权重
            if confidence > 0:
                ml_weight = min(max(confidence, 0.3), 0.7)
            else:
                ml_weight = 0.3
            wyckoff_weight = 1 - ml_weight
            
            # 综合预测
            combined_pred = ml_pred * ml_weight + wyckoff_pred * wyckoff_weight
            combined_predictions.append(combined_pred)
        
        # 更新预测结果
        ml_future_df['predicted_price'] = combined_predictions
        ml_future_df['wyckoff_phase'] = wyckoff_predictions['wyckoff_phase']
        
        return ml_future_df, confidence, _, weights

    def _predict_price_ml(self, df, future_days=30):
        """机器学习模型预测价格"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import RobustScaler
        from sklearn.model_selection import TimeSeriesSplit
        import warnings
        warnings.filterwarnings('ignore')
        
        # 准备数据
        df_pred = self.calculate_technical_indicators(df)
        features = self.create_features(df)
        
        # 获取最新日期
        last_date = pd.to_datetime(df['trade_date'].iloc[-1])
        
        # 生成未来日期
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                   periods=future_days, freq='D')
        
        # 准备预测数据
        X = df_pred[features]
        y = df_pred['close']
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 计算预测置信度
        train_pred = model.predict(X)
        confidence = 1 - np.mean(np.abs(train_pred - y) / y)
        
        # 计算预测标准差
        avg_std = np.std(train_pred - y)
        
        # 初始化预测数据框
        future_df = pd.DataFrame({
            'date': future_dates,
            'predicted_price': None
        })
        
        # 获取最新特征值
        last_features = X.iloc[-1].copy()
        
        # 逐步预测
        predictions = []
        current_features = last_features.copy()
        prev_price = df['close'].iloc[-1]
        
        for i in range(future_days):
            # 预测价格
            pred_price = model.predict([current_features])[0]
            predictions.append(pred_price)
            
            # 更新特征
            current_features = self._update_prediction_features(
                current_features, pred_price, prev_price, i+1)
            prev_price = pred_price
        
        # 添加预测结果
        future_df['predicted_price'] = predictions
        
        # 计算特征权重
        weights = dict(zip(features, model.feature_importances_))
        weights = {k: v for k, v in sorted(weights.items(), 
                                         key=lambda item: item[1], 
                                         reverse=True)}
        
        return future_df, confidence, avg_std, weights

    def analyze_buy_ratio(self, df):
        """分析主动买入情况"""
        df_tech = self.calculate_technical_indicators(df)
        recent_data = df_tech.tail(5)
        
        analysis = []
        for date, row in recent_data.iterrows():
            # 计算趋势
            if row['buy_ratio_ma5'] > df_tech['buy_ratio_ma5'].shift(1).loc[date]:
                trend = "上升"
            else:
                trend = "下降"
                
            analysis.append((
                date,
                row['buy_ratio'],
                row['buy_ratio_ma5'],
                trend
            ))
            
        return analysis

    def plot_cost_distribution(self, df):
        """绘制筹码分布"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 计算筹码分布
        price_bins, distribution = self.calculate_chip_distribution(df)
        current_price = df['close'].iloc[-1]
        
        # 计算获利盘比例
        profit_ratio = np.sum(distribution[price_bins < current_price]) / np.sum(distribution) * 100
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制筹码分布
        ax.fill_between(price_bins, distribution, alpha=0.6, color='#2ecc71', label='筹码分布')
        
        # 添加当前价格线
        ax.axvline(x=current_price, color='red', linestyle='--', label='当前价格')
        
        # 计算主要成本区域
        main_cost_price = price_bins[np.argmax(distribution)]
        ax.axvline(x=main_cost_price, color='blue', linestyle='--', label='主要成本')
        
        # 添加标题和标签
        ax.set_title(f'恒生电子筹码分布 (获利盘: {profit_ratio:.1f}%)')
        ax.set_xlabel('价格')
        ax.set_ylabel('筹码密度')
        
        # 添加网格和图例
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # 显示图形
        plt.tight_layout()
        plt.show()
        
        # 打印分析结果
        print(f"\n筹码分布分析:")
        print(f"当前价格: {current_price:.2f}")
        print(f"主力成本: {main_cost_price:.2f}")
        print(f"获利盘比例: {profit_ratio:.1f}%")
        print(f"套牢盘比例: {100-profit_ratio:.1f}%")

    def find_significant_divergences(self, df):
        """寻找显著的量价背离点"""
        df = df.copy()
        
        # 计算价格和成交量变化
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['vol'].pct_change()
        
        # 计算变化率的移动平均
        df['price_ma5'] = df['price_change'].rolling(window=5).mean()
        df['volume_ma5'] = df['volume_change'].rolling(window=5).mean()
        
        # 计算背离强度
        df['divergence_strength'] = abs(df['volume_change'] - df['price_change'])
        
        # 找出显著背离点
        significant = df[
            (df['divergence_strength'] > df['divergence_strength'].mean() + 
             df['divergence_strength'].std())
        ].copy()
        
        # 计算成交量和价格的变化率
        significant['volume_change_rate'] = significant['volume_change'] * 100
        significant['price_change_rate'] = significant['price_change'] * 100
        
        # 按背离强度排序
        significant = significant.sort_values('divergence_strength', ascending=False)
        
        # 只返回最显著的几个点
        return significant.head().sort_index()

    def add_key_events(self):
        """
        添加恒生电子的关键事件
        返回事件列表，每个事件包含日期和描述
        """
        events = [
            {'date': '2023-03-15', 'description': '发布2022年年报，营收增长22.37%'},
            {'date': '2023-08-30', 'description': '发布2023年半年报，营收增长25.12%'},
            {'date': '2023-09-20', 'description': '投资数字化转型服务商"天阙科技"'},
            {'date': '2023-11-15', 'description': '与华为签署全面合作协议'},
            {'date': '2024-01-10', 'description': '发布业绩预告，2023年净利润预增50%-70%'},
        ]
        return pd.DataFrame(events)

    def plot_stock_with_events(self):
        """
        绘制股票走势图和关键事件
        """
        # 获取数据
        df = self.get_data()
        events_df = self.add_key_events()
        events_df['date'] = pd.to_datetime(events_df['date'])

        # 创建图形和子图
        plt.style.use('seaborn')  # 使用seaborn样式
        fig = plt.figure(figsize=(20, 12))  # 增大图表尺寸
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        # 绘制股票价格
        ax1 = plt.subplot(gs[0])
        ax1.plot(df['trade_date'], df['close'], label='收盘价', color='#1f77b4', linewidth=2)
        
        # 添加事件标记
        for idx, event in events_df.iterrows():
            # 找到最接近事件日期的股价
            closest_date_idx = (df['trade_date'] - event['date']).abs().idxmin()
            closest_date = df['trade_date'].iloc[closest_date_idx]
            price = df['close'].iloc[closest_date_idx]
            
            # 绘制事件标记
            ax1.scatter(closest_date, price, color='red', s=150, zorder=5, marker='^')
            
            # 添加事件说明文字，调整文本位置和样式
            ax1.annotate(event['description'], 
                        xy=(closest_date, price),
                        xytext=(20, 20), 
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='red'),
                        fontsize=10,
                        ha='left',
                        va='bottom')

        # 设置x轴格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())  # 每月显示一个刻度
        plt.xticks(rotation=45)
        
        # 添加网格
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 设置标题和标签
        ax1.set_title('恒生电子(600570)股价走势与关键事件分析', fontsize=16, pad=20)
        ax1.set_ylabel('股价(元)', fontsize=12)
        
        # 绘制成交量
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.bar(df['trade_date'], df['vol'], color='#2ca02c', alpha=0.7, label='成交量')
        ax2.set_ylabel('成交量', fontsize=12)
        
        # 设置图例
        ax1.legend(loc='upper left', fontsize=10)
        
        # 调整布局，确保所有元素都能显示
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # 增加子图之间的间距
        
        # 显示图形
        plt.show()
        
        return fig

    def plot_prediction_analysis(self, df, future_df):
        """绘制未来30天预测分析图"""
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle('未来30天预测分析', fontsize=16)
        
        # 准备数据
        dates = pd.to_datetime(df['trade_date'])
        future_dates = future_df['date']
        all_dates = pd.concat([dates, future_dates])
        
        # 绘制历史价格和预测价格
        ax1.plot(dates, df['close'], label='历史价格', color='blue')
        ax1.plot(future_dates, future_df['predicted_price'], 
                label='预测价格', color='red', linestyle='--')
        
        # 获取威科夫分析结果
        analysis = self.analyze_wyckoff(df)
        
        # 添加支撑位和阻力位
        ax1.axhline(y=analysis['support'], color='g', linestyle='--', label='支撑位')
        ax1.axhline(y=analysis['resistance'], color='r', linestyle='--', label='阻力位')
        
        # 添加预测区间
        last_price = df['close'].iloc[-1]
        future_prices = future_df['predicted_price']
        min_price = min(future_prices.min(), last_price) * 0.95
        max_price = max(future_prices.max(), last_price) * 1.05
        ax1.fill_between(future_dates, 
                        [min_price] * len(future_dates),
                        [max_price] * len(future_dates),
                        alpha=0.1, color='gray', label='预测区间')
        
        # 标注关键信息
        ax1.text(0.02, 0.98, 
                f"当前阶段: {analysis['phase']}\n"
                f"趋势强度: {analysis['trend_strength']}/4\n"
                f"支撑位: {analysis['support']:.2f}\n"
                f"阻力位: {analysis['resistance']:.2f}",
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_title('价格预测')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # 绘制预测价格变化率
        price_changes = future_df['predicted_price'].pct_change() * 100
        colors = ['red' if x >= 0 else 'green' for x in price_changes]
        ax2.bar(future_dates[1:], price_changes[1:], color=colors)
        ax2.set_title('预测日涨跌幅 (%)')
        ax2.grid(True)
        
        # 设置x轴格式
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 调整布局
        plt.tight_layout()
        plt.show()

    def analyze(self):
        """分析股票数据"""
        df = self.get_data()
        if df is None or len(df) == 0:
            print("无法获取数据")
            return
            
        # 计算量价背离
        divergence_dates = self.check_divergence(df)
        print("\n量价背离日期:")
        print(divergence_dates[['trade_date', 'close', 'vol']])
        
        # 预测价格
        future_df, confidence, _, weights = self.predict_price(df)
        
        print(f"\n价格预测 (模型置信度: {confidence:.2%}):\n")
        print("未来30天预测价格:")
        
        # 输出威科夫分析结果
        wyckoff_analysis = self.analyze_wyckoff(df)
        print(f"\n威科夫分析:")
        print(f"当前阶段: {future_df['wyckoff_phase'].iloc[0]}")
        print(f"趋势强度: {wyckoff_analysis['trend_strength']}/4")
        print(f"支撑位: {wyckoff_analysis['support']:.2f}")
        print(f"阻力位: {wyckoff_analysis['resistance']:.2f}")
        print(f"价格位置: {wyckoff_analysis['price_position']:.2%}")
        
        # 输出预测价格
        for i in range(0, len(future_df), 5):
            print(f"日期: {future_df.iloc[i]['date'].strftime('%Y-%m-%d')}, "
                  f"预测价格: {future_df.iloc[i]['predicted_price']:.2f}")
        
        # 分析主动买入情况
        print("\n最近5天主动买入情况:")
        buy_ratio_analysis = self.analyze_buy_ratio(df)
        for date, ratio, ma5, trend in buy_ratio_analysis:
            print(f"日期: {date}, 收盘: {df.loc[date, 'close']:.2f}, "
                  f"主动买入: {ratio:.2f}%, 5日均值: {ma5:.2f}%, "
                  f"趋势: {trend}")
        
        # 输出显著的量价背离点
        print("\n显著的量价背离点:")
        significant_divergences = self.find_significant_divergences(df)
        for _, row in significant_divergences.iterrows():
            print(f"日期: {row['trade_date']}, "
                  f"收盘价: {row['close']:.2f}, "
                  f"背离强度: {row['divergence_strength']:.1f}, "
                  f"成交量变化: {row['volume_change_rate']:.1f}%, "
                  f"价格变化: {row['price_change_rate']:.1f}%")
                  
        # 显示威科夫分析图
        self.plot_wyckoff_analysis(df)

if __name__ == "__main__":
    # 恒生电子股票代码
    stock_code = '600570'
    
    analyzer = StockAnalyzer(stock_code)
    df = analyzer.get_data()
    future_df, confidence, _, weights = analyzer.predict_price(df)
    analyzer.plot_prediction_analysis(df, future_df)
