import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from datetime import datetime
from pydub import AudioSegment

from mcp.server.fastmcp import FastMCP
from dsp.dsp_utils import DSPProcessor

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
TENSOR_DIR = OUTPUT_DIR / "tensors"
PLOT_DIR = OUTPUT_DIR / "plots"

for dir_path in [AUDIO_DIR, TENSOR_DIR, PLOT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 创建 DSP 处理 MCP 服务器
mcp = FastMCP("Basic DSP Processing Tool")

# 初始化 DSP 处理器
dsp = DSPProcessor()

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

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
        output_path = TENSOR_DIR / f"stft_{get_timestamp()}.npy"
        np.save(output_path, stft_matrix)
        
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
        output_path = TENSOR_DIR / f"mfcc_{get_timestamp()}.npy"
        np.save(output_path, mfcc_features)
        
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
    """仅计算梅尔频谱图并保存为数据文件，不生成可视化图像。

    Args:
        audio_path: 音频文件路径
        n_mels: 梅尔滤波器组数量

    Returns:
        包含梅尔频谱图数据文件路径的字典
    """
    try:
        # 使用librosa加载音频
        import librosa
        
        # 加载音频
        y, sr = librosa.load(audio_path, sr=None)
        
        # 计算梅尔频谱图
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        
        # 保存梅尔频谱图数据为npy格式
        timestamp = get_timestamp()
        output_path = TENSOR_DIR / f"mel_spec_{timestamp}.npy"
        np.save(output_path, mel_spec)
        
        # 返回结果，只包含数据路径
        return {
            "success": True,
            "mel_spec_path": str(output_path),
            "n_mels": n_mels,
            "sample_rate": sr,
            "visualization": False  # 标记该函数不生成可视化
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def convert_audio_format(input_path: str, output_format: str = "wav", 
                        sample_rate: Optional[int] = None,
                        channels: Optional[int] = None,
                        bit_depth: Optional[int] = None) -> Dict[str, Any]:
    """将音频从一种格式转换为另一种格式。

    Args:
        input_path: 输入音频文件路径
        output_format: 输出格式（如 "wav", "mp3", "ogg", "flac" 等）
        sample_rate: 目标采样率（可选）
        channels: 目标通道数（可选，1=单声道，2=立体声）
        bit_depth: 目标位深度（可选，仅适用于WAV格式）

    Returns:
        包含转换后音频文件路径和音频信息的字典
    """
    try:
        # 加载音频
        audio = AudioSegment.from_file(input_path)
        
        # 应用转换参数
        if sample_rate:
            audio = audio.set_frame_rate(sample_rate)
        if channels:
            audio = audio.set_channels(channels)
        
        # 设置输出路径
        output_path = AUDIO_DIR / f"converted_{get_timestamp()}.{output_format}"
        
        # 导出音频
        if output_format.lower() == "wav" and bit_depth:
            audio = audio.set_sample_width(bit_depth // 8)
            audio.export(str(output_path), format="wav")
        else:
            audio.export(str(output_path), format=output_format)
        
        return {
            "success": True,
            "audio_path": str(output_path),
            "format": output_format,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "duration": len(audio) / 1000.0,  # 转换为秒
            "bit_depth": audio.sample_width * 8 if output_format.lower() == "wav" else None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def trim_audio(input_path: str, start_time: float, end_time: float,
               fade_in: Optional[float] = None,
               fade_out: Optional[float] = None) -> Dict[str, Any]:
    """裁剪音频文件的指定时间区间。

    Args:
        input_path: 输入音频文件路径
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        fade_in: 淡入时长（秒，可选）
        fade_out: 淡出时长（秒，可选）

    Returns:
        包含裁剪后音频文件路径和音频信息的字典
    """
    try:
        # 加载音频
        audio = AudioSegment.from_file(input_path)
        
        # 将时间转换为毫秒
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        # 确保时间在有效范围内
        start_ms = max(0, min(start_ms, len(audio)))
        end_ms = max(start_ms, min(end_ms, len(audio)))
        
        # 裁剪音频
        trimmed_audio = audio[start_ms:end_ms]
        
        # 应用淡入效果
        if fade_in is not None:
            fade_in_ms = int(fade_in * 1000)
            trimmed_audio = trimmed_audio.fade_in(fade_in_ms)
        
        # 应用淡出效果
        if fade_out is not None:
            fade_out_ms = int(fade_out * 1000)
            trimmed_audio = trimmed_audio.fade_out(fade_out_ms)
        
        # 设置输出路径
        output_path = AUDIO_DIR / f"trimmed_{get_timestamp()}.wav"
        
        # 导出音频
        trimmed_audio.export(str(output_path), format="wav")
        
        return {
            "success": True,
            "audio_path": str(output_path),
            "start_time": start_time,
            "end_time": end_time,
            "duration": len(trimmed_audio) / 1000.0,  # 转换为秒
            "sample_rate": trimmed_audio.frame_rate,
            "channels": trimmed_audio.channels,
            "fade_in": fade_in,
            "fade_out": fade_out
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def align_audio_lengths(audio_paths: List[str], target_duration: Optional[float] = None,
                       method: str = "pad", fade_duration: float = 0.1) -> Dict[str, Any]:
    """将多个音频文件对齐到相同长度。

    Args:
        audio_paths: 音频文件路径列表（至少2个）
        target_duration: 目标时长（秒，可选）。如果不指定，将使用最长音频的时长
        method: 对齐方法
            - "pad": 用静音填充到目标长度
            - "trim": 裁剪到目标长度
            - "loop": 循环重复到目标长度
            - "stretch": 拉伸或压缩到目标长度
        fade_duration: 淡入淡出时长（秒），用于平滑过渡

    Returns:
        包含对齐后音频文件路径和音频信息的字典
    """
    try:
        if len(audio_paths) < 2:
            raise ValueError("至少需要2个音频文件进行对齐")

        # 加载所有音频
        audio_segments = []
        max_duration = 0
        for path in audio_paths:
            audio = AudioSegment.from_file(path)
            audio_segments.append(audio)
            max_duration = max(max_duration, len(audio) / 1000.0)  # 转换为秒

        # 如果没有指定目标时长，使用最长音频的时长
        if target_duration is None:
            target_duration = max_duration

        # 将目标时长转换为毫秒
        target_ms = int(target_duration * 1000)
        fade_ms = int(fade_duration * 1000)

        # 对齐所有音频
        aligned_audios = []
        for i, audio in enumerate(audio_segments):
            current_duration = len(audio)
            
            if method == "pad":
                # 用静音填充
                if current_duration < target_ms:
                    padding = AudioSegment.silent(duration=target_ms - current_duration)
                    aligned = audio + padding
                else:
                    aligned = audio[:target_ms]
                    
            elif method == "trim":
                # 裁剪到目标长度
                aligned = audio[:target_ms]
                
            elif method == "loop":
                # 循环重复
                if current_duration < target_ms:
                    repeats = int(np.ceil(target_ms / current_duration))
                    aligned = audio * repeats
                    aligned = aligned[:target_ms]
                else:
                    aligned = audio[:target_ms]
                    
            elif method == "stretch":
                # 拉伸或压缩
                ratio = target_ms / current_duration
                aligned = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * ratio)
                }).set_frame_rate(audio.frame_rate)
                aligned = aligned[:target_ms]
                
            else:
                raise ValueError(f"不支持的对齐方法: {method}")

            # 应用淡入淡出效果
            if fade_ms > 0:
                aligned = aligned.fade_in(fade_ms).fade_out(fade_ms)

            # 保存对齐后的音频
            output_path = AUDIO_DIR / f"aligned_{i}_{get_timestamp()}.wav"
            aligned.export(str(output_path), format="wav")
            aligned_audios.append(str(output_path))

        return {
            "success": True,
            "aligned_paths": aligned_audios,
            "target_duration": target_duration,
            "method": method,
            "fade_duration": fade_duration,
            "sample_rate": audio_segments[0].frame_rate,
            "channels": audio_segments[0].channels
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # 初始化并运行服务器
    mcp.run() 