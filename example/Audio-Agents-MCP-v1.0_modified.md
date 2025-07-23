# Audio-Agents-MCP v1.0

![Audio-Agents-MCP](assets/audio_agents_logo.png)

Audio-Agents-MCP is an audio processing toolkit based on the Model Context Protocol (MCP), providing comprehensive functionality from basic DSP processing to advanced audio generation. The project adopts a modular design, separating basic DSP functions from advanced audio features for easy extension and maintenance.

## Features

### 1. Basic DSP Functions (dsp_processor)
- Audio loading and saving
- Resampling
- Feature extraction (STFT, MFCC, Mel spectrogram)
- Audio effects (reverb, fade in/out)
- Audio mixing

### 2. Advanced Audio Features (audio_processor)
- Voice editing (using VoiceCraft)
- Text-to-speech (using Spark TTS)
- Audio processing (using Kimi-Audio)
- Music generation (using InspireMusic/NotaGen)

## System Requirements

- Python 3.10+
- Dependencies (automatically installed via requirements):
  - python-dotenv
  - mcp[cli]
  - openai
  - colorama
  - torch>=2.0.0
  - torchaudio>=2.0.0
  - librosa>=0.10.0
  - soundfile>=0.12.1
  - transformers>=4.30.0

## Installation Steps

1. **Clone the repository:**
   ```bash
   git clone git@github.com:your-username/audio-agents-mcp.git
   cd audio-agents-mcp
   ```

2. **Set up virtual environment:**
   ```bash
   # Install uv (if not already installed)
   pip install uv

   # Create virtual environment and install dependencies
   uv venv .venv --python=3.10

   # Activate virtual environment
   # macOS/Linux
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or use uv for faster installation
   uv pip install -r requirements.txt
   ```

4. **Configure environment:**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` file to add necessary configurations:
     ```
     LLM_MODEL_NAME=your_llm_model_name
     LLM_BASE_URL=your_llm_api_url
     LLM_API_KEY=your_llm_api_key
     OLLAMA_MODEL_NAME=your_ollama_model_name
     OLLAMA_BASE_URL=your_ollama_api_url
     ```

## Starting the Server

1. **Start all MCP servers:**
   ```bash
   python scripts/start_servers.py
   ```

## Usage Examples

### 1. Basic DSP Functions

#### Audio Loading and Resampling
```python
from mcp import MCPClient
import base64

async def process_audio():
    client = MCPClient()
    with open("input.wav", "rb") as f:
        audio_data = base64.b64encode(f.read()).decode()
    
    # Resample
    response = await client.call(
        server="dsp_processor",
        action="resample_audio",
        parameters={
            "audio_data": audio_data,
            "orig_sr": 44100,
            "target_sr": 16000
        }
    )
    
    if response.success:
        with open("resampled.wav", "wb") as f:
            f.write(base64.b64decode(response.result["audio_data"]))
```

#### Adding Audio Effects
```python
async def add_effects():
    client = MCPClient()
    with open("input.wav", "rb") as f:
        audio_data = base64.b64encode(f.read()).decode()
    
    # Add reverb
    response = await client.call(
        server="dsp_processor",
        action="add_reverb",
        parameters={
            "audio_data": audio_data,
            "room_scale": 0.8
        }
    )
    
    if response.success:
        with open("reverb.wav", "wb") as f:
            f.write(base64.b64decode(response.result["audio_data"]))
```

### 2. Advanced Audio Features

#### Voice Editing
```python
async def edit_voice():
    client = MCPClient()
    with open("input.wav", "rb") as f:
        audio_data = base64.b64encode(f.read()).decode()
    
    response = await client.call(
        server="audio_processor",
        action="edit_voice",
        parameters={
            "audio_data": audio_data,
            "target_text": "New text",
            "start_time": 1.5,
            "end_time": 3.0
        }
    )
    
    if response.success:
        with open("edited.wav", "wb") as f:
            f.write(base64.b64decode(response.result["audio_data"]))
```

#### Text-to-Speech
```python
async def generate_speech():
    client = MCPClient()
    response = await client.call(
        server="audio_processor",
        action="generate_speech",
        parameters={
            "text": "Hello, world",
            "speaker_id": "speaker1",
            "speed": 1.2
        }
    )
    
    if response.success:
        with open("speech.wav", "wb") as f:
            f.write(base64.b64decode(response.result["audio_data"]))
```

#### Music Generation
```python
async def generate_music():
    client = MCPClient()
    response = await client.call(
        server="audio_processor",
        action="generate_music",
        parameters={
            "prompt": "Happy piano music",
            "duration": 30.0,
            "style": "classical"
        }
    )
    
    if response.success:
        with open("music.wav", "wb") as f:
            f.write(base64.b64decode(response.result["audio_data"]))
```

## 洛神赋全文

昔有佳人，
召南抱素怀，
临川望瑶台。
翩若惊鸿，
婉若游龙。
荣曜秋菊，
华茂春松。
仿佛兮若轻云之蔽月，
飘飘兮若流风之回雪。

远而望之，
皎若太阳升朝霞；
迫而察之，
灼若芙蕖出渌波。

翩若惊鸿，婉若游龙，
荣曜秋菊，华茂春松。

朝游江渚，暮宿云霞。
行止俨然，若思天上人。

忽逢桃花林，夹岸数百步，中无杂树，芳草鲜美，落英缤纷。
渔人甚异之，复前行，欲穷其林。

林尽水源，便得一山，山有小口，仿佛若有光。
便舍船，从口入。

初极狭，才通人；复行数十步，豁然开朗。
土地平旷，屋舍俨然，有良田、美池、桑竹之属。
阡陌交通，鸡犬相闻。
其中往来种作，男女衣着，悉如外人。
黄发垂髫，怡然自乐。

见渔人，乃大惊，问所从来。
具答之，便要还家，设酒杀鸡作食。
村中闻有此人，咸来问讯。

自云先世避秦时乱，率妻子邑人来此绝境，不复出焉，遂与外人间隔。
问今是何世，乃不知有汉，无论魏、晋。
此人一一为具言所闻，皆叹惋。

余人各复延至其家，皆出酒食。
停数日，辞去。
道出桃花源，便得一丘。

桃花源记。

——晋·陶渊明