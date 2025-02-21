import torch
from llm_demo import SimpleLLM

def load_model(model_path, vocab_size=1000, embed_dim=256, num_heads=8, ff_dim=512, num_layers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLLM(vocab_size, embed_dim, num_heads, ff_dim, num_layers)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model, device

def generate_text(model, prompt_text, vocab_size=1000, max_length=50):
    device = next(model.parameters()).device
    # 将提示文本转换为输入格式
    input_ids = torch.tensor([[ord(c) % vocab_size for c in prompt_text]]).to(device)
    
    # 生成文本
    with torch.no_grad():
        generated = model.generate(input_ids, max_length=max_length, temperature=0.7)
    
    # 将生成的token转换回文本
    generated_text = ""
    for token in generated[0]:
        generated_text += chr(token.item())
    
    return generated_text

def main():
    # 加载训练好的模型
    model_path = 'llm_model.pth'
    model, device = load_model(model_path)
    
    # 测试不同的提示文本
    test_prompts = [
        "人工智能",
        "深度学习",
        "语言模型",
        "神经网络"
    ]
    
    print("模型测试结果：")
    print("-" * 50)
    for prompt in test_prompts:
        print(f"\n输入提示：{prompt}")
        generated = generate_text(model, prompt)
        print(f"生成结果：{generated}")
        print("-" * 50)

if __name__ == "__main__":
    main()