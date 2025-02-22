# Screenshot to Vue Code Converter

这个项目可以将截图转换为Vue代码。它使用GPT-4 Vision API来分析截图，并生成对应的Vue组件代码，包含完整的Tailwind CSS样式。

## 功能特点

- 支持拖放上传图片
- 实时预览上传的图片
- 生成响应式Vue组件代码
- 使用Tailwind CSS进行样式设计
- 支持一键复制生成的代码

## 安装和运行

### 后端设置

1. 进入后端目录：
```bash
cd backend
```

2. 创建并编辑环境变量文件：
```bash
echo "OPENAI_API_KEY=your-api-key" > .env
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 运行后端服务：
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 前端设置

1. 进入前端目录：
```bash
cd frontend
```

2. 安装依赖：
```bash
npm install
```

3. 运行开发服务器：
```bash
npm run dev
```

4. 在浏览器中访问：
```
http://localhost:5173
```

## 使用方法

1. 打开网页后，你可以通过点击上传区域或直接拖放图片到该区域来上传截图
2. 上传完成后，系统会自动分析图片并生成对应的Vue组件代码
3. 你可以点击"复制代码"按钮来复制生成的代码
4. 将代码粘贴到你的Vue项目中即可使用

## 注意事项

- 确保你有有效的OpenAI API密钥，并且有访问GPT-4 Vision API的权限
- 生成的代码使用Tailwind CSS进行样式设计，请确保你的项目中已经配置了Tailwind CSS
- 上传的图片会被自动压缩以适应API的限制
- 生成的代码是完整的Vue组件，包含了所有必要的导入语句和样式

## 技术栈

- 后端：FastAPI + Python
- 前端：Vue 3 + Vite
- 样式：Tailwind CSS
- AI：GPT-4 Vision API
