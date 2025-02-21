import torch
import torch.nn as nn
from llm_demo import SimpleLLM
from torch.utils.data import Dataset, DataLoader

class SimpleTextDataset(Dataset):
    def __init__(self, texts, vocab_size, max_length=50):
        self.data = texts
        self.vocab_size = vocab_size
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx].strip()  # 移除换行符
        # 将文本转换为 token
        tokens = [ord(c) % self.vocab_size for c in text]
        
        # 确保长度一致：截断或填充
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        # 创建输入和目标
        input_ids = torch.tensor(tokens[:-1])
        target_ids = torch.tensor(tokens[1:])
        
        return input_ids, target_ids

def train_model(model, train_loader, num_epochs, device, learning_rate=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # 前向传播
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')

def load_training_data():
    # 这里可以从文件加载大量文本数据
    sample_texts = []
    
    # 从文件加载数据
    try:
        with open('data/training_texts.txt', 'r', encoding='utf-8') as f:
            sample_texts = f.readlines()
    except FileNotFoundError:
        # 如果文件不存在，生成一些示例数据
        base_texts = [
            "人工智能正在快速发展。",
            "深度学习模型需要大量训练数据。",
            "语言模型可以理解和生成文本。",
            "神经网络由多个层组成。",
            "注意力机制是一个重要的突破。"
        ]
        # 通过组合生成更多数据
        for t1 in base_texts:
            for t2 in base_texts:
                sample_texts.append(t1 + t2)
                
    return sample_texts

def main():
    # 模型参数
    vocab_size = 1000
    embed_dim = 256
    num_heads = 8
    ff_dim = 512
    num_layers = 4
    batch_size = 32
    num_epochs = 10
    
    # 示例训练数据
    sample_texts = [
        "这是一个示例文本。",
        "用于训练语言模型。",
        "我们需要更多的训练数据。"
        # 实际使用时需要大量真实数据
    ]
    
    # 加载更多训练数据
    training_texts = load_training_data()
    print(f"加载了 {len(training_texts)} 条训练数据")
    
    # 创建数据集和数据加载器
    dataset = SimpleTextDataset(training_texts, vocab_size)
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4  # 使用多进程加载数据
    )
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = SimpleLLM(vocab_size, embed_dim, num_heads, ff_dim, num_layers)
    model = model.to(device)
    
    # 训练模型
    train_model(model, train_loader, num_epochs, device)
    
    # 保存模型
    torch.save(model.state_dict(), 'llm_model.pth')

if __name__ == "__main__":
    main()