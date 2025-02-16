import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os

class SmartKnowledgeBase:
    def __init__(self, collection_name="trading_docs"):
        # 初始化 Chroma
        self.client = chromadb.Client()
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction()
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        
        # 初始化 DeepSeek 客户端
        self.ai_client = OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ.get("DEEPSEEK_API_KEY")
        )

    def add_documents(self, documents, ids=None):
        """添加文档到知识库"""
        if ids is None:
            ids = [str(i) for i in range(len(documents))]
            
        contents = [doc['content'] for doc in documents]
        metadatas = [
            {
                'title': doc['title'],
                'category': doc.get('metadata', {}).get('category', ''),
                'year': str(doc.get('metadata', {}).get('year', ''))
            }
            for doc in documents
        ]
        
        self.collection.add(
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )

    def smart_search(self, query, n_results=2):
        """智能搜索并生成回答"""
        # 首先搜索相关文档
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # 构建上下文
        context = "\n\n".join([
            f"文档：{doc}\n标题：{meta['title']}"
            for doc, meta in zip(
                results['documents'][0],
                results['metadatas'][0]
            )
        ])
        
        # 使用 DeepSeek 生成回答
        prompt = f"""基于以下文档内容回答问题：

文档内容：
{context}

问题：{query}

请提供简洁明了的回答。"""

        response = self.ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return {
            'answer': response.choices[0].message.content,
            'sources': results
        }

# 使用示例
if __name__ == "__main__":
    # 创建知识库实例
    kb = SmartKnowledgeBase()
    
    # 添加示例文档
    documents = [
        {
            'title': '比特币简介',
            'content': '比特币是一种去中心化的数字货币，由中本聪在2008年首次提出。它使用区块链技术确保交易安全性。',
            'metadata': {'category': 'cryptocurrency', 'year': 2008}
        },
        {
            'title': '以太坊简介',
            'content': '以太坊是一个开源的区块链平台，支持智能合约功能。它不仅是一种加密货币，还是一个去中心化应用平台。',
            'metadata': {'category': 'cryptocurrency', 'year': 2015}
        },
        {
            'title': '区块链技术',
            'content': '区块链是一种分布式账本技术，具有去中心化、不可篡改等特点。它是比特币等加密货币的基础技术。',
            'metadata': {'category': 'technology', 'year': 2008}
        }
    ]
    
    kb.add_documents(documents)
    
    # 智能搜索示例
    query = "解释区块链技术如何确保比特币的安全性？"
    result = kb.smart_search(query)
    
    # 打印结果
    print(f"\n问题: {query}")
    print("\nAI 回答:")
    print(result['answer'])
    print("\n参考文档:")
    for i, (doc, metadata) in enumerate(zip(
        result['sources']['documents'][0],
        result['sources']['metadatas'][0]
    )):
        print(f"\n文档 {i+1}:")
        print(f"标题: {metadata['title']}")
        print(f"内容: {doc}")