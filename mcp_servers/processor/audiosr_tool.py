#!/usr/bin/env python3
"""
AudioSR Tool: Versatile Audio Super-Resolution at Scale
直接使用AudioSR Python API进行音频超分辨率处理

Based on: https://github.com/haoheliu/versatile_audio_super_resolution
AudioSR可以将任何采样率的音频提升到48kHz高质量输出
支持所有类型的音频：音乐、语音、环境声音等
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

def main():
    """AudioSR工具主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == '--params_file':
        # 从参数文件读取参数
        params_file = sys.argv[2]
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        result = process_audiosr(**params)
        print(json.dumps(result))
    else:
        # 命令行模式
        parser = argparse.ArgumentParser(description="AudioSR Tool: Audio Super-Resolution")
        parser.add_argument("--audio_file", required=True, help="Input audio file path")
        parser.add_argument("--output_file", required=False, help="Output audio file path")
        parser.add_argument("--model_name", choices=["basic", "speech"], default="basic", 
                          help="Model to use for super-resolution")
        parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM sampling steps")
        parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        
        args = parser.parse_args()
        result = process_audiosr(
            audio_file=args.audio_file,
            output_file=args.output_file,
            model_name=args.model_name,
            ddim_steps=args.ddim_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
        print(json.dumps(result, indent=2))

def process_audiosr(
    audio_file: str,
    output_file: Optional[str] = None,
    model_name: str = "basic",
    ddim_steps: int = 50,
    guidance_scale: float = 3.5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    使用AudioSR进行音频超分辨率处理
    
    Args:
        audio_file: 输入音频文件路径
        output_file: 输出音频文件路径（可选）
        model_name: 使用的模型名称（basic或speech）
        ddim_steps: DDIM采样步数
        guidance_scale: 引导尺度
        seed: 随机种子
        
    Returns:
        处理结果字典
    """
    try:
        # 导入必要的库
        import torch
        import librosa
        import soundfile as sf
        from audiosr import build_model, super_resolution
        
        # 验证输入文件
        if not os.path.exists(audio_file):
            return {
                "success": False,
                "error": f"输入音频文件不存在: {audio_file}"
            }
        
        # 生成输出文件路径
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_name = Path(audio_file).stem
            output_file = f"audiosr_{model_name}_{input_name}_{timestamp}.wav"
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 构建模型 - device会在build_model中自动处理
        print(f"加载AudioSR模型: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        audiosr = build_model(model_name=model_name, device=device)
        
        # 获取原始音频信息
        print(f"加载音频文件信息: {audio_file}")
        original_waveform, original_sr = librosa.load(audio_file, sr=None)
        
        # 执行超分辨率处理 - 使用正确的API调用方式
        print("执行音频超分辨率处理...")
        try:
            enhanced_waveform = super_resolution(
                audiosr,
                audio_file,  # 直接传文件路径
                seed=seed,
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA内存不足，切换到CPU模式重新处理...")
                # 重新构建模型使用CPU
                audiosr = build_model(model_name=model_name, device="cpu")
                enhanced_waveform = super_resolution(
                    audiosr,
                    audio_file,
                    seed=seed,
                    ddim_steps=ddim_steps,
                    guidance_scale=guidance_scale
                )
                device = "cpu"  # 更新设备记录
            else:
                raise e
        
        # 确保waveform是正确的1D数组格式
        if enhanced_waveform.ndim > 1:
            enhanced_waveform = enhanced_waveform.squeeze()  # 移除多余的维度
        
        # 保存结果
        print(f"保存处理后的音频: {output_file}")
        sf.write(output_file, enhanced_waveform, 48000)  # AudioSR输出固定为48kHz
        
        # 获取音频信息
        duration = len(enhanced_waveform) / 48000
        file_size = os.path.getsize(output_file)
        
        return {
            "success": True,
            "output_file": output_file,
            "model_used": model_name,
            "original_sr": int(original_sr),
            "output_sr": 48000,
            "duration": round(duration, 2),
            "file_size": file_size,
            "device_used": device,
            "ddim_steps": ddim_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "message": f"AudioSR处理完成，音频已提升到48kHz高质量输出"
        }
        
    except ImportError as e:
        return {
            "success": False,
            "error": f"导入AudioSR库失败: {str(e)}。请确保已安装audiosr库。"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"AudioSR处理失败: {str(e)}"
        }

def AudioSRTool(
    audio_file: str,
    output_file: Optional[str] = None,
    model_name: str = "basic"
) -> Dict[str, Any]:
    """
    AudioSR工具函数 - 供MCP启动器调用
    
    Args:
        audio_file: 输入音频文件路径
        output_file: 输出音频文件路径（可选）
        model_name: 使用的模型名称（basic或speech）
        
    Returns:
        处理结果字典
    """
    return process_audiosr(
        audio_file=audio_file,
        output_file=output_file,
        model_name=model_name,
        ddim_steps=50,  # 默认DDIM步数
        guidance_scale=3.5,  # 默认引导尺度
        seed=42  # 默认随机种子
    )

if __name__ == "__main__":
    main()



