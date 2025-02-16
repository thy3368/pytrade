import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client

class TradeTimeseriesDB:
    def __init__(self, db_path="trades.db"):
        # 初始化 SQLite 数据库
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # 创建交易表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                timestamp DATETIME,
                symbol TEXT,
                price REAL,
                volume REAL,
                is_buyer_maker INTEGER,
                PRIMARY KEY (timestamp, symbol)
            )
        ''')
        self.conn.commit()
        
        # 初始化 Binance 客户端
        self.binance_client = Client()
        
    def store_trades(self, symbol="BTCUSDT", hours=1):
        """存储交易数据到数据库"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        trades = self.binance_client.get_aggregate_trades(
            symbol=symbol,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000)
        )
        
        # 准备数据
        data = []
        for trade in trades:
            data.append({
                'timestamp': datetime.fromtimestamp(trade['T']/1000),
                'symbol': symbol,
                'price': float(trade['p']),
                'volume': float(trade['q']),
                'is_buyer_maker': int(trade['m'])
            })
        
        # 转换为 DataFrame 并存储
        df = pd.DataFrame(data)
        df.to_sql('trades', self.conn, if_exists='append', index=False)
        
        return len(trades)

    def query_trades(self, symbol="BTCUSDT", hours=1):
        """查询交易数据"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        query = '''
        SELECT * FROM trades 
        WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp DESC
        '''
        
        df = pd.read_sql_query(
            query, 
            self.conn,
            params=(symbol, start_time, end_time)
        )
        
        return df

    def get_price_stats(self, symbol="BTCUSDT", hours=1):
        """获取价格统计信息"""
        df = self.query_trades(symbol, hours)
        
        if df.empty:
            return {
                'symbol': symbol,
                'period': f'{hours}小时',
                'data': []
            }
        
        stats = {
            'symbol': symbol,
            'period': f'{hours}小时',
            'avg_price': df['price'].mean(),
            'max_price': df['price'].max(),
            'min_price': df['price'].min(),
            'total_volume': df['volume'].sum(),
            'trade_count': len(df)
        }
        
        return stats

    def __del__(self):
        """关闭数据库连接"""
        self.conn.close()

# 使用示例
if __name__ == "__main__":
    db = TradeTimeseriesDB()
    
    # 存储交易数据
    print("存储 BTC/USDT 交易数据...")
    count = db.store_trades("BTCUSDT", 1)
    print(f"已存储 {count} 条交易记录")
    
    # 查询数据
    print("\n查询最近交易数据...")
    trades_df = db.query_trades("BTCUSDT", 1)
    print("\n最近交易记录:")
    print(trades_df.head())
    
    # 获取统计信息
    print("\n价格统计信息:")
    stats = db.get_price_stats("BTCUSDT", 1)
    print(stats)