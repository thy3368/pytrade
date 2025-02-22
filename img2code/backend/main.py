import os
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai
from PIL import Image
import io

# 加载环境变量
load_dotenv()

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def image_to_base64(image_file: bytes) -> str:
    """将图片转换为base64编码"""
    return base64.b64encode(image_file).decode('utf-8')

def compress_image(image_bytes: bytes, max_size: int = 1024) -> bytes:
    """压缩图片到指定大小"""
    img = Image.open(io.BytesIO(image_bytes))
    
    # 如果图片是PNG格式且有透明通道，转换为RGB
    if img.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    
    # 计算新的尺寸
    ratio = max_size / max(img.size)
    if ratio < 1:
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # 保存压缩后的图片
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85, optimize=True)
    return buffer.getvalue()

@app.post("/api/convert")
async def convert_image(file: UploadFile = File(...), framework: Optional[str] = "vue"):
    """将上传的图片转换为代码"""
    try:
        # 读取并压缩图片
        contents = await file.read()
        compressed_contents = compress_image(contents)
        base64_image = image_to_base64(compressed_contents)
        
        # 准备提示词
        prompt = f"""
        You are an expert web developer. Please convert this image into a responsive {framework.upper()} component.
        Use Tailwind CSS for styling.
        The component should be:
        1. Pixel perfect match to the image
        2. Fully responsive
        3. Use semantic HTML
        4. Follow accessibility best practices
        5. Include all necessary imports
        
        Return ONLY the complete code without any explanations.
        The code should be ready to use in a {framework.upper()} project.
        """
        
        # 调用GPT-4 Vision API
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096
        )
        
        # 返回生成的代码
        return {"code": response.choices[0].message.content}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
