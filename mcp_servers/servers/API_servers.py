import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import inspect
import tempfile
import requests
from typing import Optional, Dict, Any, List, Union, Literal

from gradio_client import Client, handle_file

from mcp.server.fastmcp import FastMCP

# 初始化MCP服务器
mcp = FastMCP("使用API调用tool而不是本地部署的示例，完成后需要在tool_descriptions.json中添加工具的使用说明")



@mcp.tool()
def image2music_api(
    image_path: str,
    output_dir: str = "outputs/music",
    model: Literal['ACE Step', 'AudioLDM-2', 'Riffusion', 'Mustango', 'Stable Audio Open'] = 'ACE Step'

) -> str:
    """Generate music from an image using the image-to-music-v2 model.
    
    Args:
        image_path (str): Path to the local image file.
        output_dir (str, optional): Path to the output directory. Defaults to "outputs/music".
        model (Literal, optional): Model to use for music generation. 
            Options: 'ACE Step', 'AudioLDM-2', 'Riffusion', 'Mustango', 'Stable Audio Open'. 
            Defaults to 'ACE Step'.
    
    Returns:
        str: Path to the generated audio file.
    """
    # Create a client connection to the Hugging Face space
    client = Client("fffiloni/image-to-music-v2")
    
    # Ensure the image path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    

    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"music_{timestamp}.wav"
    
    try:
        # Call the API with the image and parameters
        result = client.predict(
            image_in=handle_file(image_path),  # 正确处理图片文件
            chosen_model=model,                # 选择模型
            api_name="/infer"                  # 正确的API名称
        )
        
        # 处理提示词
        prompt_data = result[0]
        if isinstance(prompt_data, dict) and 'value' in prompt_data:
            prompt = prompt_data['value']
            print(f"Generated inspirational prompt: {prompt}")
        else:
            print(f"Generated data: {prompt_data}")
        
        # 处理音频文件
        audio_url = result[1]
        
        # 使用临时文件下载音频
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close()
        
        # 如果是URL，则下载
        if isinstance(audio_url, str) and (audio_url.startswith('http://') or audio_url.startswith('https://')):
            response = requests.get(audio_url)
            if response.status_code == 200:
                with open(temp_file.name, 'wb') as f:
                    f.write(response.content)
            else:
                raise Exception(f"Failed to download audio file: {response.status_code}")
        # 如果已经是本地文件路径
        elif isinstance(audio_url, str) and os.path.exists(audio_url):
            with open(audio_url, 'rb') as f_in:
                with open(temp_file.name, 'wb') as f_out:
                    f_out.write(f_in.read())
        else:
            raise Exception(f"Unexpected audio result format: {type(audio_url)}")
        
        # 复制到最终输出位置
        with open(temp_file.name, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                f_out.write(f_in.read())
        
        # 清理临时文件
        try:
            os.unlink(temp_file.name)
        except:
            pass
            
        return str(output_file)
    except Exception as e:
        import traceback
        return f"Error generating music: {str(e)}\n{traceback.format_exc()}"

if __name__ == "__main__":
    mcp.run()



