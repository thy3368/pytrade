import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 前馈网络
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        
    def forward(self, x, mask=None):
        # 自注意力层
        x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)
        attn_output = attn_output.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        x = x.transpose(0, 1)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 前馈网络
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # 解码器层
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"Expected input with 2 dimensions (batch_size, seq_len), got {x.dim()}")
            
        # 生成因果注意力掩码
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)
        
        # 词嵌入和位置编码
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # 通过解码器层
        for layer in self.layers:
            x = layer(x, mask)
            
        # 输出层
        return self.output_layer(x)

    @torch.no_grad()
    def generate(self, input_ids, max_length=50, temperature=1.0):
        self.eval()
        current_ids = input_ids
        
        for _ in range(max_length):
            outputs = self(current_ids)
            logits = outputs[:, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            if next_token.item() == 0:  # 假设0为结束标记
                break
                
        return current_ids

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

def main():
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型参数
    vocab_size = 1000
    embed_dim = 256
    num_heads = 8
    ff_dim = 512
    num_layers = 4
    
    # 创建模型
    model = SimpleLLM(vocab_size, embed_dim, num_heads, ff_dim, num_layers)
    model = model.to(device)
    
    # 测试数据
    batch_size = 2
    seq_len = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # 前向传播测试
    output = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("\n模型总参数量:", sum(p.numel() for p in model.parameters()))
    
    # 生成测试
    input_prompt = torch.tensor([[1, 2, 3]]).to(device)  # 示例输入
    generated = model.generate(input_prompt)
    print("\n生成结果形状:", generated.shape)

if __name__ == "__main__":
    main()