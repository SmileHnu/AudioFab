# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

```bash
# Environment setup
conda activate AudioFab

# Start complete system (backend servers + frontend)
python scripts/start_all.py

# Start only MCP backend servers
python scripts/start_servers.py

# Run tests
scripts/unittest.sh              # Run all tests
scripts/unittest.sh test/chat/test_session.py  # Run specific test

# Validate configuration
scripts/check.sh
```

## Architecture Overview

AudioFab is an intelligent audio agent framework built on the Model-Context-Protocol (MCP) architecture. It enables natural language interaction with complex audio processing tasks through intelligent tool selection and execution.

### Core Components

- **MCP Client**: Handles user interactions and coordinates between LLM Planner and MCP Server
- **LLM Planner**: Task planning, tool selection, and response generation using OpenAI/Ollama
- **MCP Server**: Manages 9 specialized services with isolated dependency environments
- **Tool Kits**: Audio processing tools organized by functionality

### MCP Services (9 specialized servers)

1. **markdown_servers**: File I/O for text formats (md, txt, json)
2. **dsp_servers**: Basic digital signal processing (STFT, MFCC, format conversion)
3. **audio_servers**: Comprehensive audio processing (loading, resampling, effects)
4. **tensor_servers**: PyTorch/NumPy tensor operations
5. **tool_query_servers**: Tool discovery and intelligent search
6. **FunTTS_mcp_servers**: Speech recognition, synthesis, enhancement, emotion analysis
7. **music_mcp_servers**: Music generation, lyrics-to-song, audio-driven video
8. **Audioseparator_mcp_servers**: Audio separation and reconstruction
9. **API_servers**: External API integrations (Hugging Face, ModelScope)

### Key Tool Categories

- **Speech Processing**: Whisper, FunASR, CosyVoice2, SparkTTS, VoiceCraft
- **Music Generation**: DiffRhythm, ACE-Step, YuE, MusicGen, AudioGen
- **Audio Enhancement**: ClearVoice, audio separation, super-resolution
- **Visual Audio**: Hallo2 (talking head videos), AudioX (video generation)

## Configuration

### Environment Variables (.env)
```
LLM_API_KEY=your_openai_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4-turbo-preview
OLLAMA_MODEL_NAME=your_ollama_model
OLLAMA_BASE_URL=http://localhost:11434
MARKDOWN_FOLDER_PATH=/path/to/files
RESULT_FOLDER_PATH=/path/to/results
```

### Server Configuration (mcp_servers/servers_config.json)
Update Python interpreter paths for your environment:
```json
{
  "mcpServers": {
    "markdown_servers": {
      "command": "/your/python/path",
      "args": ["mcp_servers/servers/markdown_servers.py"]
    }
  }
}
```

## Development Workflow

1. **Setup**: Create conda environment from `environment.yml` or `environment-lock.yml`
2. **Configuration**: Update `.env` and `servers_config.json` with actual paths
3. **Validation**: Run `scripts/check.sh` to verify configuration
4. **Testing**: Use `scripts/unittest.sh` for test-driven development
5. **Development**: Start with `python scripts/start_all.py` for full system access

## Testing Structure

- **Unit tests**: `test/` directory with focused component testing
- **Integration tests**: Full system testing with MCP servers
- **Test utilities**: Mock-based testing for MCP client interactions

## Important Directories

- `mcp_chatbot/`: Core MCP client and LLM integration
- `mcp_servers/`: All MCP server implementations and configurations
- `mcp_servers/processor/`: Individual tool processors
- `models/`: Third-party model integrations (TIGER, VoiceCraft, etc.)
- `example/`: Usage examples and demos
- `scripts/`: System startup and utility scripts