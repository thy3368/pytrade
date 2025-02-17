import streamlit as st
from datetime import datetime

def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def add_message(role, content):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

def main():
    st.title("交易助手")
    
    # 初始化聊天历史
    init_chat_history()
    
    # 侧边栏配置
    st.sidebar.header("设置")
    if st.sidebar.button("清空聊天记录"):
        st.session_state.messages = []
    
    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(f"{message['timestamp']} - {message['content']}")
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题"):
        # 添加用户消息
        add_message("user", prompt)
        
        # 这里添加你的交易分析逻辑
        response = f"收到您的问题：{prompt}\n目前这只是一个演示回复。"
        
        # 添加助手回复
        with st.chat_message("assistant"):
            st.write(f"{datetime.now().strftime('%H:%M:%S')} - {response}")
            add_message("assistant", response)

if __name__ == "__main__":
    main()