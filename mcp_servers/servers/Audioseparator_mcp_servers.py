import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import inspect
import tempfile
from typing import Optional, Dict, Any, List, Union, Literal

from mcp.server.fastmcp import FastMCP

# 初始化MCP服务器
mcp = FastMCP("音频分离服务：集成AudioSep、Audio-Separator、TIGER和AudioSR模型")

# 工具环境配置
TOOL_ENV_CONFIG = {
    "AudioSepTool": {
        "python_path": "/home/chengz/anaconda3/envs/AudioSep/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "Audiosep_processor.py")
    },
    "AudioSeparatorTool": {
        "python_path": "/home/chengz/anaconda3/envs/voicecraft2/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "audio_separator_processor.py")
    },
    "TIGERSpeechSeparationTool": {
        "python_path": "/home/chengz/anaconda3/envs/Tiger/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "TIGER_speech_separation_processor.py")
    },
    "AudioSRTool": {
        "python_path": "/home/qianshuaix/miniconda3/envs/audiosr/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor"  / "audiosr_tool.py")
    }
}

# 启动器脚本路径
LAUNCHER_SCRIPT = str(Path(__file__).parent.parent / "mcp_tool_launcher.py")

# 创建临时目录（如果不存在）
TEMP_DIR = Path(tempfile.gettempdir()) / "mcp_temp"
TEMP_DIR.mkdir(exist_ok=True)

def execute_tool_in_env(tool_name, **kwargs):
    """在特定环境中执行工具
    
    Args:
        tool_name: 要执行的工具名称
        **kwargs: 传递给工具的参数
        
    Returns:
        工具执行结果
    """
    # 检查工具配置是否存在
    if tool_name not in TOOL_ENV_CONFIG:
        return {"success": False, "error": f"工具 '{tool_name}' 没有环境配置"}
    
    tool_config = TOOL_ENV_CONFIG[tool_name]
    
    # 创建临时参数文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_params_file = TEMP_DIR / f"{tool_name}_{timestamp}.json"
    
    # 将参数写入临时文件
    with open(temp_params_file, 'w') as f:
        json.dump(kwargs, f)
    
    try:
        # 构建命令 - 使用固定的启动器脚本
        cmd = [
            tool_config["python_path"],
            LAUNCHER_SCRIPT,
            "--tool_name", tool_name,
            "--module_path", tool_config["script_path"],
            "--params_file", str(temp_params_file)
        ]
        
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 清理临时文件
        try:
            os.remove(temp_params_file)
        except:
            pass  # 忽略清理错误
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"工具执行失败: {result.stderr}"
            }
        
        # 解析结果
        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": f"无法解析工具结果: {result.stdout}"
            }
            
    except Exception as e:
        # 清理临时文件
        try:
            os.remove(temp_params_file)
        except:
            pass  # 忽略清理错误
            
        return {"success": False, "error": str(e)}

# 直接定义工具而不导入

@mcp.tool()
def AudioSepTool(
    # Input content
    audio_file: str,
    text: str,
    
    # Output configuration
    output_file: Optional[str] = None,
    
    # Processing options
    use_chunk: bool = False,
    
    # Computing device
    device: str = "cuda"
) -> Dict[str, Any]:
    """Separate specific sounds from an audio file using textual descriptions.
    
    Args:
        audio_file: Path to the input audio file to process
        text: Textual description of the sound to separate (e.g., "piano playing", "dog barking")
        output_file: Path to save the separated audio file. If not provided, a default path will be used
        use_chunk: Whether to use chunked processing for memory efficiency with longer audio files
        device: Computing device to use (cuda or cpu)
        
    Returns:
        Dictionary containing the separation results and metadata
    """
    return execute_tool_in_env("AudioSepTool", 
                               audio_file=audio_file,
                               text=text,
                               output_file=output_file,
                               use_chunk=use_chunk,
                               device=device)

@mcp.tool()
def AudioSeparatorTool(
    # Input content
    audio_path: str,
    
    # Model selection
    model_name: str = "UVR-MDX-NET-Inst_HQ_3.onnx",
    
    # Output options
    output_dir: Optional[str] = None,
    output_format: str = "WAV"
) -> Dict[str, Any]:
    """Separate audio into different stems (vocals, instrumental, drums, bass, etc.).
    
    Args:
        audio_path: Path to the input audio file
        model_name: Model to use for separation (UVR-MDX-NET-Inst_HQ_3.onnx or htdemucs_6s.yaml)
        output_dir: Directory to save separated stems
        output_format: Format of output audio files (WAV, MP3, FLAC, etc.)
    
    Returns:
        Dictionary containing the separation results and paths to output files
    """
    return execute_tool_in_env("AudioSeparatorTool", 
                              audio_path=audio_path,
                              model_name=model_name,
                              output_dir=output_dir,
                              output_format=output_format)

@mcp.tool()
def TIGERSpeechSeparationTool(
    # Input audio
    audio_path: str,
    
    # Output configuration
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Separate speech from audio mixtures.
    
    Args:
        audio_path: Path to the input audio file containing mixed speech to separate.
        output_dir: Directory to save the separated speech files. Each speaker 
                    will be saved as a separate WAV file.
    
    Returns:
        Dictionary containing paths to all separated audio files
    """
    return execute_tool_in_env("TIGERSpeechSeparationTool", 
                              audio_path=audio_path,
                              output_dir=output_dir)

@mcp.tool()
def AudioSRTool(
    # Input audio
    audio_file: str,
    
    # Output configuration  
    output_file: Optional[str] = None,
    
    # Model selection
    model_name: str = "basic"
) -> Dict[str, Any]:
    """Enhance audio quality through super-resolution (upscale to 48kHz).
    
    AudioSR can improve audio quality for any type of audio (music, speech, environmental sounds)
    by upscaling to 48kHz high-quality output regardless of input sample rate. The device will
    be automatically selected (CUDA if available, otherwise CPU).
    
    Args:
        audio_file: Path to the input audio file to enhance
        output_file: Path to save the enhanced audio file. If not provided, 
                    a timestamped file will be created automatically
        model_name: Model to use for enhancement - "basic" for general audio or "speech" for speech audio
    
    Returns:
        Dictionary containing the enhanced audio file path and processing metadata
    """
    return execute_tool_in_env("AudioSRTool", 
                              audio_file=audio_file,
                              output_file=output_file,
                              model_name=model_name)

if __name__ == "__main__":
    # 启动MCP服务器
    mcp.run()
