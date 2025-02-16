import chromadb
from chromadb.utils import embedding_functions

class SimpleChromaDB:
    def __init__(self, collection_name="trading_docs"):
        # 初始化 Chroma 客户端
        self.client = chromadb.Client()
        # 使用默认的 sentence-transformers 嵌入函数
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction()
        # 创建或获取集合
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_documents(self, documents, ids=None):
        """添加文档到知识库"""
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
            
        # 提取文档内容和元数据
        contents = [doc['content'] for doc in documents]
        metadatas = [
            {
                'title': doc['title'],
                'category': doc.get('metadata', {}).get('category', ''),
                'year': str(doc.get('metadata', {}).get('year', ''))
            }
            for doc in documents
        ]
        
        # 添加到集合
        self.collection.add(
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query, n_results=2):
        """搜索相关文档"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

# 使用示例
if __name__ == "__main__":
    # 创建知识库实例
    db = SimpleChromaDB()
    
    # 准备示例文档
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
    
    # 添加文档
    db.add_documents(documents)
    
    # 搜索示例
    query = "去中心化技术"
    results = db.search(query)
    
    # 打印结果
    print(f"\n查询: {query}")
    print("\n相关文档:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n文档 {i+1}:")
        print(f"标题: {metadata['title']}")
        print(f"内容: {doc}")
        print(f"相似度: {1 - distance:.4f}")
        print(f"类别: {metadata['category']}")
        print(f"年份: {metadata['year']}")