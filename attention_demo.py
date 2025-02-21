import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 定义线性变换层
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 线性变换并分头
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 调整维度顺序
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重
        out = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 合并多头注意力的结果
        out = out.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        out = out.view(batch_size, seq_len, self.embed_dim)
        
        return self.out_linear(out)

def main():
    # 测试代码
    batch_size = 2
    seq_len = 4
    embed_dim = 8
    num_heads = 2
    
    # 创建模型
    attention = SelfAttention(embed_dim, num_heads)
    
    # 创建输入数据
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # 前向传播
    output = attention(x)
    
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("\n注意力输出示例:")
    print(output[0, 0, :])  # 打印第一个批次，第一个位置的输出向量

if __name__ == "__main__":
    main()