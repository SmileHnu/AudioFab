import subprocess
import time
import os
import sys

# 启动 MCP 后端服务器（后台新窗口）
backend = subprocess.Popen(
    [sys.executable, "scripts/start_servers.py"],
    creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
)

# 等待后端服务启动
time.sleep(1)

# # 启动 Streamlit 前端
subprocess.run([
    "streamlit", "run", "example/chatbot_streamlit/app2.py",
    "--server.port=8080"
]) 


# 启动 Streamlit 前端
# subprocess.run([
#     "streamlit", "run", "example/chatbot_streamlit/app2.py",
#     "--server.address=0.0.0.0",
#     "--server.port=8080",
#     "--browser.serverAddress=10.64.68.74",
#     "--browser.serverPort=8080 ",
#     "--server.enableStaticServing=true",
#     "--server.enableCORS=false",
#     "--server.enableXsrfProtection=false"

# ])