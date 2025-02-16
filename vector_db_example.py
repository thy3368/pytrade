from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class SimpleVectorDB:
    def __init__(self):
        # 初始化向量模型
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # 模型输出维度
        # 初始化FAISS索引
        self.index = faiss.IndexFlatL2(self.dimension)
        # 存储原始数据
        self.data = []
        
    def add_trade_data(self, trade_info):
        """添加交易数据到向量数据库"""
        # 将交易信息转换为文本
        text = f"价格:{trade_info['price']} 数量:{trade_info['volume']}"
        # 计算向量
        vector = self.model.encode([text])[0]
        # 添加到FAISS
        self.index.add(np.array([vector]).astype('float32'))
        # 保存原始数据
        self.data.append(trade_info)
        
    def search_similar(self, query_trade, k=5):
        """搜索相似的交易"""
        # 将查询转换为向量
        text = f"价格:{query_trade['price']} 数量:{query_trade['volume']}"
        query_vector = self.model.encode([text])[0]
        
        # 搜索最相似的k个结果
        distances, indices = self.index.search(
            np.array([query_vector]).astype('float32'), k
        )
        
        # 返回结果
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # 确保找到了结果
                results.append({
                    'trade_info': self.data[idx],
                    'distance': float(dist)
                })
        return results

# 使用示例
if __name__ == "__main__":
    # 创建向量数据库实例
    db = SimpleVectorDB()
    
    # 添加一些示例数据
    sample_trades = [
        {'price': 50000, 'volume': 1.5, 'timestamp': datetime.now()},
        {'price': 49800, 'volume': 0.8, 'timestamp': datetime.now()},
        {'price': 50200, 'volume': 2.0, 'timestamp': datetime.now()},
        {'price': 49900, 'volume': 1.2, 'timestamp': datetime.now()},
    ]
    
    # 添加数据到数据库
    for trade in sample_trades:
        db.add_trade_data(trade)
    
    # 搜索相似交易
    query = {'price': 50100, 'volume': 1.4}
    results = db.search_similar(query, k=2)
    
    # 打印结果
    print("\n相似交易查询结果:")
    for r in results:
        print(f"价格: {r['trade_info']['price']}, "
              f"数量: {r['trade_info']['volume']}, "
              f"相似度距离: {r['distance']:.4f}")