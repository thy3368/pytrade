from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
import chromadb
from chromadb.utils import embedding_functions

class BinanceDataKB:
    def __init__(self, collection_name="trading_data"):
        # 初始化 Binance 客户端
        self.binance_client = Client()
        
        # 初始化 Chroma
        self.client = chromadb.Client()
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction()
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def get_recent_trades(self, symbol="BTCUSDT", hours=1):
        """获取最近的交易数据"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # 获取交易数据
        trades = self.binance_client.get_aggregate_trades(
            symbol=symbol,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000)
        )

        print(trades)
        # 获取K线数据作为买卖价格参考
        klines = self.binance_client.get_historical_klines(
            symbol,
            Client.KLINE_INTERVAL_1MINUTE,
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # 转换为 DataFrame
        df = pd.DataFrame(trades)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['T'], unit='ms')
            df['price'] = df['p'].astype(float)
            df['volume'] = df['q'].astype(float)
            df['is_buyer_maker'] = df['m'].astype(bool)
            
            # 处理K线数据
            kline_df = pd.DataFrame(klines, columns=['kline_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                                                   'quote_volume', 'trades_count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'])
            kline_df['timestamp'] = pd.to_datetime(kline_df['kline_time'], unit='ms')
            kline_df['bid'] = kline_df['low'].astype(float)  # 最低价作为买一价
            kline_df['ask'] = kline_df['high'].astype(float)  # 最高价作为卖一价
            
            # 合并数据
            df = pd.merge_asof(df.sort_values('timestamp'), 
                            kline_df[['timestamp', 'bid', 'ask']].sort_values('timestamp'),
                            on='timestamp',
                            direction='nearest')
            
            # 计算主动买卖指标
            df['is_buy_initiative'] = df['price'] >= df['ask']
            df['is_sell_initiative'] = df['price'] <= df['bid']
    
        return df

    def store_trade_analysis(self, symbol="BTCUSDT", hours=1):
        """分析并存储交易数据"""
        df = self.get_recent_trades(symbol, hours)
        if df.empty:
            return "无交易数据"
            
        # 计算基本统计信息
        analysis = {
            'symbol': symbol,
            'period': f"最近{hours}小时",
            'avg_price': df['price'].mean(),
            'max_price': df['price'].max(),
            'min_price': df['price'].min(),
            'total_volume': df['volume'].sum(),
            'trade_count': len(df),
            'buy_initiative_count': df['is_buyer_maker'].sum(),
            'sell_initiative_count': df['is_sell_initiative'].sum(),
            'buy_initiative_ratio': (df['is_buy_initiative'].sum() / len(df)) * 100,
            'timestamp': datetime.now()
        }
        
        # 生成分析文本
        content = (
            f"{symbol} {analysis['period']}交易分析：\n"
            f"平均价格: {analysis['avg_price']:.2f}\n"
            f"最高价格: {analysis['max_price']:.2f}\n"
            f"最低价格: {analysis['min_price']:.2f}\n"
            f"总交易量: {analysis['total_volume']:.4f}\n"
            f"交易次数: {analysis['trade_count']}\n"
            f"主动买入次数: {analysis['buy_initiative_count']}\n"
            f"主动卖出次数: {analysis['sell_initiative_count']}\n"
            f"主动买入比例: {analysis['buy_initiative_ratio']:.2f}%"
        )
        
        # 存储到知识库
        self.collection.add(
            documents=[content],
            metadatas=[{
                'symbol': symbol,
                'hours': str(hours),
                'timestamp': analysis['timestamp'].isoformat()
            }],
            ids=[f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"]
        )
        
        return content

    def query_analysis(self, query, n_results=2):
        """查询历史分析数据"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

# 使用示例
if __name__ == "__main__":
    # 创建实例
    kb = BinanceDataKB()
    
    # 获取并存储 BTC/USDT 的交易分析
    print("获取 BTC/USDT 交易数据...")
    analysis = kb.store_trade_analysis("BTCUSDT", hours=1)
    print("\n分析结果:")
    print(analysis)
    
    # 获取并存储 ETH/USDT 的交易分析
    print("\n获取 ETH/USDT 交易数据...")
    analysis = kb.store_trade_analysis("ETHUSDT", hours=1)
    print("\n分析结果:")
    print(analysis)


    analysis = kb.store_trade_analysis("TRUMPUSDT", hours=1)


    # 查询示例
    print("\n查询历史分析...")
    query = "BTC 价格分析"
    results = kb.query_analysis(query)
    
    print(f"\n查询: {query}")
    print("\n相关记录:")
    for i, (doc, metadata) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0]
    )):
        print(f"\n记录 {i+1}:")
        print(f"币对: {metadata['symbol']}")
        print(f"时间: {metadata['timestamp']}")
        print(f"内容:\n{doc}")