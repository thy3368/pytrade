from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict

class SimpleKnowledgeBase:
    def __init__(self):
        # 初始化向量模型
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        # 初始化FAISS索引
        self.index = faiss.IndexFlatL2(self.dimension)
        # 存储文档
        self.documents = []
        
    def add_document(self, title: str, content: str, metadata: Dict = None):
        """添加文档到知识库"""
        # 计算文档向量
        vector = self.model.encode([content])[0]
        # 添加到FAISS
        self.index.add(np.array([vector]).astype('float32'))
        # 保存文档
        self.documents.append({
            'title': title,
            'content': content,
            'metadata': metadata or {}
        })
        
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """搜索相关文档"""
        # 计算查询向量
        query_vector = self.model.encode([query])[0]
        
        # 搜索最相似的文档
        distances, indices = self.index.search(
            np.array([query_vector]).astype('float32'), k
        )
        
        # 返回结果
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                results.append({
                    'document': self.documents[idx],
                    'similarity_score': 1 / (1 + float(dist))  # 转换距离为相似度分数
                })
        return results

# 使用示例
if __name__ == "__main__":
    # 创建知识库实例
    kb = SimpleKnowledgeBase()
    
    # 添加示例文档
    documents = [
        {
            'title': '比特币简介',
            'content': '比特币是一种去中心化的数字货币，由中本聪在2008年首次提出。',
            'metadata': {'category': 'cryptocurrency', 'year': 2008}
        },
        {
            'title': '以太坊简介',
            'content': '以太坊是一个开源的区块链平台，支持智能合约功能。',
            'metadata': {'category': 'cryptocurrency', 'year': 2015}
        },
        {
            'title': '区块链技术',
            'content': '区块链是一种分布式账本技术，具有去中心化、不可篡改等特点。',
            'metadata': {'category': 'technology', 'year': 2008}
        }
    ]
    
    # 添加文档到知识库
    for doc in documents:
        kb.add_document(doc['title'], doc['content'], doc['metadata'])
    
    # 搜索示例
    query = "去中心化"
    results = kb.search(query, k=2)
    
    # 打印结果
    print(f"\n查询: {query}")
    print("\n相关文档:")
    for r in results:
        print(f"\n标题: {r['document']['title']}")
        print(f"内容: {r['document']['content']}")
        print(f"相似度: {r['similarity_score']:.4f}")
        print(f"元数据: {r['document']['metadata']}")