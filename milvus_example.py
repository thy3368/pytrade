from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
import numpy as np
from datetime import datetime
from binance.client import Client
import pandas as pd

class TradingVectorDB:
    def __init__(self, collection_name="trade_vectors"):
        # 连接到 Milvus
        connections.connect(host='localhost', port='19530')
        
        self.collection_name = collection_name
        self.dim = 8  # 向量维度：价格、数量、时间等特征
        
        # 创建集合
        self._create_collection()
        
        # 初始化 Binance 客户端
        self.binance_client = Client()
    
    def _create_collection(self):
        """创建 Milvus 集合"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="symbol", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="price", dtype=DataType.FLOAT),
            FieldSchema(name="volume", dtype=DataType.FLOAT),
            FieldSchema(name="features", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        
        schema = CollectionSchema(fields=fields, description="交易数据向量存储")
        self.collection = Collection(name=self.collection_name, schema=schema)
        
        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="features", index_params=index_params)
    
    def _create_feature_vector(self, trade_data):
        """创建特征向量"""
        # 简单特征工程示例
        price = float(trade_data['price'])
        volume = float(trade_data['volume'])
        timestamp = int(trade_data['timestamp'].timestamp())
        
        # 构建特征向量（示例特征）
        features = np.array([
            price,
            volume,
            timestamp % 86400,  # 一天内的秒数
            timestamp % 3600,   # 小时内的秒数
            np.log(price),
            np.log(volume),
            price * volume,     # 成交额
            float(trade_data['is_buyer_maker'])
        ], dtype=np.float32)
        
        return features
    
    def store_trades(self, symbol="BTCUSDT", limit=100):
        """存储交易数据"""
        # 获取交易数据
        trades = self.binance_client.get_recent_trades(symbol=symbol, limit=limit)
        
        # 准备数据
        entities = {
            "symbol": [],
            "timestamp": [],
            "price": [],
            "volume": [],
            "features": []
        }
        
        for trade in trades:
            trade_data = {
                'price': float(trade['price']),
                'volume': float(trade['qty']),
                'timestamp': datetime.fromtimestamp(trade['time']/1000),
                'is_buyer_maker': trade['isBuyerMaker']
            }
            
            features = self._create_feature_vector(trade_data)
            
            entities["symbol"].append(symbol)
            entities["timestamp"].append(int(trade_data['timestamp'].timestamp()))
            entities["price"].append(trade_data['price'])
            entities["volume"].append(trade_data['volume'])
            entities["features"].append(features.tolist())
        
        # 插入数据
        self.collection.insert(entities)
        return len(trades)
    
    def search_similar_trades(self, trade_data, top_k=5):
        """搜索相似交易"""
        # 生成查询向量
        query_vector = self._create_feature_vector(trade_data)
        
        # 执行搜索
        self.collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="features",
            param=search_params,
            limit=top_k,
            output_fields=["symbol", "timestamp", "price", "volume"]
        )
        
        # 处理结果
        similar_trades = []
        for hits in results:
            for hit in hits:
                similar_trades.append({
                    "symbol": hit.entity.get('symbol'),
                    "timestamp": datetime.fromtimestamp(hit.entity.get('timestamp')),
                    "price": hit.entity.get('price'),
                    "volume": hit.entity.get('volume'),
                    "distance": hit.distance
                })
        
        return similar_trades

# 使用示例
if __name__ == "__main__":
    db = TradingVectorDB()
    
    # 存储交易数据
    print("存储交易数据...")
    count = db.store_trades("BTCUSDT", limit=100)
    print(f"已存储 {count} 条交易记录")
    
    # 搜索相似交易
    query_trade = {
        'price': 50000.0,
        'volume': 1.5,
        'timestamp': datetime.now(),
        'is_buyer_maker': True
    }
    
    print("\n搜索相似交易...")
    similar_trades = db.search_similar_trades(query_trade, top_k=3)
    
    print("\n相似交易结果:")
    for trade in similar_trades:
        print(f"\n价格: {trade['price']}")
        print(f"数量: {trade['volume']}")
        print(f"时间: {trade['timestamp']}")
        print(f"相似度距离: {trade['distance']}")