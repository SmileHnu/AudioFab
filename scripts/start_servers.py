import asyncio
import os
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime


import asyncio
import sys



# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/audio_agents_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Audio-Agents-MCP')

def ensure_logs_directory():
    """确保日志目录存在"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

def load_server_config():
    """加载服务器配置"""
    config_path = Path("mcp_servers/servers_config.json")


    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载服务器配置失败: {str(e)}")
        raise

def start_server(server_name: str, config: dict):
    """启动单个服务器"""
    server_config = config["mcpServers"][server_name]
    command = server_config["command"]
    args = server_config["args"]
    
    # 替换路径占位符
    args = [arg.replace("/path/to/your", str(Path.cwd())) for arg in args]
    
    logger.info(f"正在启动 {server_name}...")
    process = subprocess.Popen(
        [command] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return process

def monitor_server_output(server_name: str, process: subprocess.Popen):
    """监控服务器输出"""
    while True:
        # 读取输出
        stdout = process.stdout.readline()
        if stdout:
            logger.info(f"[{server_name}] {stdout.strip()}")
        
        # 检查是否出错
        if process.poll() is not None:
            stderr = process.stderr.read()
            if stderr:
                logger.error(f"[{server_name}] Error: {stderr}")
            logger.error(f"[{server_name}] 服务器意外停止")
            return False
        
        # 短暂休眠以避免CPU过载
        asyncio.sleep(0.1)

async def main():
    """启动所有服务器"""
    try:
        # 确保日志目录存在
        ensure_logs_directory()
        
        # 加载配置
        config = load_server_config()
        
        # 启动所有服务器
        processes = []
        for server_name in config["mcpServers"]:
            process = start_server(server_name, config)
            processes.append((server_name, process))
        
        logger.info("\n所有服务器已启动。按 Ctrl+C 停止所有服务器。\n")
        
        # 监控所有服务器的输出
        monitoring_tasks = [
            asyncio.create_task(monitor_server_output(server_name, process))
            for server_name, process in processes
        ]
        
        try:
            # 等待所有监控任务完成
            await asyncio.gather(*monitoring_tasks)
        except KeyboardInterrupt:
            logger.info("\n正在停止所有服务器...")
            for server_name, process in processes:
                process.terminate()
                process.wait()
                logger.info(f"[{server_name}] 服务器已停止")
        
    except Exception as e:
        logger.error(f"启动服务器时发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("程序已终止")
    except Exception as e:
        logger.error(f"程序异常终止: {str(e)}")
        exit(1) 