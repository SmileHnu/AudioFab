import asyncio
import os
from pathlib import Path

from mcp_chatbot import Configuration, MCPClient, ChatSession
from mcp_chatbot.llm import create_llm_client

async def main():
    """示例：使用音频处理功能"""
    # 加载配置
    config = Configuration()
    server_config = config.load_config("mcp_servers/servers_config.json")
    
    # 创建所有可用的客户端
    clients = [
        MCPClient(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    
    # 创建 LLM 客户端
    llm_client = create_llm_client(provider="openai", config=config)
    
    # 创建聊天会话
    chat_session = ChatSession(clients, llm_client)
    
    try:
        # 初始化会话
        await chat_session.initialize()
        
        # 示例1：文本转语音
        print("\n1. 文本转语音示例")
        response = await chat_session.send_message(
            "请将文本'你好，这是一个测试。'转换为语音",
            show_workflow=True
        )
        print(f"结果: {response}")
        
        # 示例2：音频处理
        print("\n2. 音频处理示例")
        input_audio = "example/data/input.wav"
        if os.path.exists(input_audio):
            response = await chat_session.send_message(
                f"请对音频文件 {input_audio} 进行标准化和压缩处理",
                show_workflow=True
            )
            print(f"结果: {response}")
        
        # 示例3：音乐生成
        print("\n3. 音乐生成示例")
        response = await chat_session.send_message(
            "请生成一首30秒的欢快电子音乐",
            show_workflow=True
        )
        print(f"结果: {response}")
        
    finally:
        # 清理资源
        await chat_session.cleanup_clients()

if __name__ == "__main__":
    asyncio.run(main()) 