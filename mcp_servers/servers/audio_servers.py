import os
from pathlib import Path
import io
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import http.server
import socketserver
import threading
import shutil
import webbrowser
from urllib.parse import urljoin
import socket
import librosa
import librosa.display

from mcp.server.fastmcp import FastMCP
from dsp.dsp_utils import DSPProcessor


# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
TENSOR_DIR = OUTPUT_DIR / "tensors"
PLOT_DIR = OUTPUT_DIR / "plots"

for dir_path in [AUDIO_DIR, TENSOR_DIR, PLOT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 创建音频处理 MCP 服务器
mcp = FastMCP("Advanced Audio Processing Tool")

# 初始化处理器
dsp = DSPProcessor()

# 创建临时服务器目录
TEMP_SERVER_DIR = OUTPUT_DIR / "temp_server"
TEMP_SERVER_DIR.mkdir(parents=True, exist_ok=True)

# 全局变量存储服务器信息
server_info = {
    "is_running": False,
    "port": 8000,
    "server": None,
    "thread": None
}

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_local_ip() -> str:
    """获取本机的IP地址"""
    try:
        # 创建一个临时socket连接来获取本机IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"  # 如果获取失败，返回本地回环地址

@mcp.tool()
def load_audio(audio_path: str, target_sr: Optional[int] = None) -> Dict[str, Any]:
    """加载音频数据。

    Args:
        audio_path: 音频文件路径
        target_sr: 目标采样率（可选）

    Returns:
        包含音频文件路径和采样率的字典
    """
    try:
        # 加载音频
        audio, sr = dsp.load_audio(audio_path, target_sr)
        
        # 保存处理后的音频
        output_path = AUDIO_DIR / f"processed_{get_timestamp()}.wav"
        dsp.save_audio(audio, output_path, sr)
        
        return {
            "success": True,
            "audio_path": str(output_path),
            "sample_rate": sr
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def resample_audio(audio_path: str, orig_sr: int, target_sr: int) -> Dict[str, Any]:
    """重采样音频。

    Args:
        audio_path: 音频文件路径
        orig_sr: 原始采样率
        target_sr: 目标采样率

    Returns:
        包含重采样后的音频文件路径和采样率的字典
    """
    try:
        # 加载音频
        audio, _ = dsp.load_audio(audio_path)
        
        # 重采样
        resampled_audio = dsp.resample(audio, orig_sr, target_sr)
        
        # 保存重采样后的音频
        output_path = AUDIO_DIR / f"resampled_{get_timestamp()}.wav"
        dsp.save_audio(resampled_audio, output_path, target_sr)
        
        return {
            "success": True,
            "audio_path": str(output_path),
            "sample_rate": target_sr
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def compute_stft(audio_path: str, n_fft: int = 2048, hop_length: int = 512) -> Dict[str, Any]:
    """计算短时傅里叶变换。

    Args:
        audio_path: 音频文件路径
        n_fft: FFT窗口大小
        hop_length: 帧移

    Returns:
        包含STFT矩阵文件路径的字典
    """
    try:
        # 加载音频
        audio, _ = dsp.load_audio(audio_path)
        
        # 计算STFT
        stft_matrix = dsp.stft(audio, n_fft, hop_length)
        
        # 保存STFT矩阵
        output_path = TENSOR_DIR / f"stft_{get_timestamp()}.pth"
        torch.save(stft_matrix, output_path)
        
        return {
            "success": True,
            "stft_path": str(output_path),
            "n_fft": n_fft,
            "hop_length": hop_length
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def compute_mfcc(audio_path: str, n_mfcc: int = 13) -> Dict[str, Any]:
    """计算MFCC特征。

    Args:
        audio_path: 音频文件路径
        n_mfcc: MFCC系数数量

    Returns:
        包含MFCC特征文件路径的字典
    """
    try:
        # 加载音频
        audio, _ = dsp.load_audio(audio_path)
        
        # 计算MFCC
        mfcc_features = dsp.mfcc(audio, n_mfcc)
        
        # 保存MFCC特征
        output_path = TENSOR_DIR / f"mfcc_{get_timestamp()}.pth"
        torch.save(mfcc_features, output_path)
        
        return {
            "success": True,
            "mfcc_path": str(output_path),
            "n_mfcc": n_mfcc
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def compute_mel_spectrogram(audio_path: str, n_mels: int = 128) -> Dict[str, Any]:
    """计算梅尔频谱图并生成可视化图像。

    Args:
        audio_path: 音频文件路径
        n_mels: 梅尔滤波器组数量

    Returns:
        包含梅尔频谱图可视化文件路径的字典
    """
    try:
        # 使用librosa加载音频
        y, sr = librosa.load(audio_path, sr=None)
        
        # 计算梅尔频谱图
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        
        # 生成时间戳，确保文件名唯一
        timestamp = get_timestamp()
        
        # 保存梅尔频谱图数据（副产品）
        data_path = TENSOR_DIR / f"mel_spec_{timestamp}.pth"
        torch.save(mel_spec, data_path)
        
        # 定义图像保存路径（主要产品）
        plot_path = PLOT_DIR / f"mel_spec_{timestamp}.png"
        
        # 创建高质量的梅尔频谱图可视化
        plt.figure(figsize=(12, 8))
        
        # 转换为分贝刻度，便于更好的可视化
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 使用librosa的display来生成更专业的specshow
        librosa.display.specshow(
            mel_spec_db, 
            y_axis='mel', 
            x_axis='time',
            sr=sr,
            cmap='magma'  # 使用更好的颜色映射方案，如'magma'或'viridis'
        )
        
        # 添加颜色条，显示分贝值
        cbar = plt.colorbar(format='%+2.0f dB')
        cbar.set_label('Intensity (dB)')
        
        # 设置图表标题和轴标签
        plt.title('Mel Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        
        # 调整布局并保存高分辨率图像
        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=300)
        plt.close()
        
        return {
            "success": True,
            "mel_spec_path": str(data_path),
            "plot_path": str(plot_path),
            "n_mels": n_mels,
            "sample_rate": sr,
            "visualization": True  # 标记该函数生成了可视化
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def add_reverb(audio_path: str, room_scale: float = 0.8) -> Dict[str, Any]:
    """添加混响效果。

    Args:
        audio_path: 音频文件路径
        room_scale: 房间大小因子（0-1）

    Returns:
        包含处理后音频文件路径的字典
    """
    try:
        # 加载音频
        audio, sr = dsp.load_audio(audio_path)
        
        # 添加混响
        reverb_audio = dsp.add_reverb(audio, room_scale)
        
        # 保存处理后的音频
        output_path = AUDIO_DIR / f"reverb_{get_timestamp()}.wav"
        dsp.save_audio(reverb_audio, output_path, sr)
        
        return {
            "success": True,
            "audio_path": str(output_path),
            "sample_rate": sr
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def mix_audio(audio_paths: List[str], weights: Optional[List[float]] = None) -> Dict[str, Any]:
    """混合多个音频。

    Args:
        audio_paths: 音频文件路径列表
        weights: 混合权重列表（可选）

    Returns:
        包含混合后音频文件路径的字典
    """
    try:
        # 加载所有音频
        audio_list = []
        sr = None
        for audio_path in audio_paths:
            audio, current_sr = dsp.load_audio(audio_path)
            audio_list.append(audio)
            if sr is None:
                sr = current_sr
        
        # 混合音频
        mixed_audio = dsp.mix_audio(audio_list, weights)
        
        # 保存混合后的音频
        output_path = AUDIO_DIR / f"mixed_{get_timestamp()}.wav"
        dsp.save_audio(mixed_audio, output_path, sr)
        
        return {
            "success": True,
            "audio_path": str(output_path),
            "sample_rate": sr
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def apply_fade(audio_path: str, fade_duration: float) -> Dict[str, Any]:
    """应用淡入淡出效果。

    Args:
        audio_path: 音频文件路径
        fade_duration: 淡入淡出时长（秒）

    Returns:
        包含处理后音频文件路径的字典
    """
    try:
        # 加载音频
        audio, sr = dsp.load_audio(audio_path)
        
        # 应用淡入淡出
        faded_audio = dsp.apply_fade(audio, fade_duration)
        
        # 保存处理后的音频
        output_path = AUDIO_DIR / f"faded_{get_timestamp()}.wav"
        dsp.save_audio(faded_audio, output_path, sr)
        
        return {
            "success": True,
            "audio_path": str(output_path),
            "sample_rate": sr
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def start_file_server(port: int = 8000):
    """启动一个简单的HTTP服务器来托管文件"""
    class FileHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(TEMP_SERVER_DIR), **kwargs)
        
        def log_message(self, format, *args):
            # 自定义日志格式，显示访问信息
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {format%args}")

    with socketserver.TCPServer(("0.0.0.0", port), FileHandler) as httpd:
        server_info["server"] = httpd
        server_info["is_running"] = True
        print(f"Server started on port {port}")
        print(f"Local URL: http://localhost:{port}")
        print(f"Network URL: http://{get_local_ip()}:{port}")
        httpd.serve_forever()

@mcp.tool()
def serve_local_audio(audio_path: str, port: int = 8000) -> Dict[str, Any]:
    """将本地音频文件转换为可访问的URL。

    Args:
        audio_path: 本地音频文件路径
        port: HTTP服务器端口号（默认8000）

    Returns:
        包含音频URL的字典
    """
    try:
        # 确保音频文件存在
        if not os.path.exists(audio_path):
            return {
                "success": False,
                "error": f"Audio file not found: {audio_path}"
            }

        # 如果服务器已经在运行，先停止它
        if server_info["is_running"]:
            server_info["server"].shutdown()
            server_info["thread"].join()
            server_info["is_running"] = False

        # 复制音频文件到临时服务器目录
        filename = os.path.basename(audio_path)
        temp_path = TEMP_SERVER_DIR / filename
        shutil.copy2(audio_path, temp_path)

        # 启动服务器
        server_info["port"] = port
        server_info["thread"] = threading.Thread(
            target=start_file_server,
            args=(port,),
            daemon=True
        )
        server_info["thread"].start()

        # 获取本机IP地址
        local_ip = get_local_ip()
        
        # 构建URLs
        local_url = f"http://localhost:{port}/{filename}"
        network_url = f"http://{local_ip}:{port}/{filename}"

        return {
            "success": True,
            "local_url": local_url,
            "network_url": network_url,
            "local_path": str(temp_path),
            "port": port,
            "ip_address": local_ip
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def stop_audio_server() -> Dict[str, Any]:
    """停止音频文件上传服务器，并释放资源。

    Returns:
        操作结果字典
    """
    try:
        if server_info["is_running"]:
            server_info["server"].shutdown()
            server_info["thread"].join()
            server_info["is_running"] = False
            
            # 清理临时文件
            for file in TEMP_SERVER_DIR.glob("*"):
                file.unlink()
            
            return {
                "success": True,
                "message": "Server stopped successfully"
            }
        else:
            return {
                "success": True,
                "message": "Server was not running"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # 初始化并运行服务器
    mcp.run() 