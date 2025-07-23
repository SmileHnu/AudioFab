import asyncio
import json
import os
import re
import sys
import base64
from io import BytesIO
import shutil
import time
from pathlib import Path
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image
import uuid
import numpy as np
import tempfile

# 尝试导入sounddevice和soundfile用于音频录制
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_RECORDING_AVAILABLE = True
except (ImportError, OSError) as e:
    AUDIO_RECORDING_AVAILABLE = False
    print(f"音频录制不可用: {str(e)}")
    
# 确保声明变量，避免未定义错误
if 'AUDIO_RECORDING_AVAILABLE' not in globals():
    AUDIO_RECORDING_AVAILABLE = False

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

# Create media storage directories
MEDIA_DIR = os.path.join(PROJECT_ROOT, "user_media")
IMAGES_DIR = os.path.join(MEDIA_DIR, "images")
AUDIO_DIR = os.path.join(MEDIA_DIR, "audio")
VIDEO_DIR = os.path.join(MEDIA_DIR, "video")

# Add output directories
# OUTPUT_DIR = "/home/chengz/LAMs/mcp_chatbot-audio/output"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
OUTPUT_AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
OUTPUT_VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")
OUTPUT_PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Create directories if they don't exist
for directory in [MEDIA_DIR, IMAGES_DIR, AUDIO_DIR, VIDEO_DIR, 
                 OUTPUT_DIR, OUTPUT_AUDIO_DIR, OUTPUT_VIDEO_DIR, OUTPUT_PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Assuming these imports exist and work correctly
from mcp_chatbot import Configuration, MCPClient  # noqa: E402
from mcp_chatbot.chat import ChatSession  # noqa: E402
from mcp_chatbot.llm import create_llm_client  # noqa: E402
from mcp_chatbot.mcp.mcp_tool import MCPTool  # noqa: E402



# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Audio-Agent Ultra", layout="wide")

# --- Streamlit Logo Configuration ---
# st.logo(
#     os.path.join(PROJECT_ROOT, "assets", "mcp_chatbot_logo.png"),
#     size="large",
# )


logo_path = os.path.join(PROJECT_ROOT, "assets", "mcp_chatbot_logo.png")
if os.path.exists(logo_path):
    st.image(logo_path, width=200)  # 使用 st.image 显示 logo



# Add custom CSS

st.title("🤖 AudioFab 🤖")
st.caption(
    "A chatbot that uses the Model Context Protocol (MCP) to interact with tools."
)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "openai"

if "chatbot_config" not in st.session_state:
    st.session_state.chatbot_config = Configuration()

if "mcp_tools_cache" not in st.session_state:
    st.session_state.mcp_tools_cache = {}

# Add state for chat session and config tracking
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "session_config_hash" not in st.session_state:
    st.session_state.session_config_hash = None
if "active_mcp_clients" not in st.session_state:
    st.session_state.active_mcp_clients = []  # Track active clients outside stack
if "mcp_client_stack" not in st.session_state:
    st.session_state.mcp_client_stack = None  # Store the stack itself
if "history_messages" not in st.session_state:
    st.session_state.history_messages = []

# Media storage in session state
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = {}
if "uploaded_audio" not in st.session_state:
    st.session_state.uploaded_audio = {}
if "uploaded_videos" not in st.session_state:
    st.session_state.uploaded_videos = {}
if "recorded_audio" not in st.session_state:
    st.session_state.recorded_audio = {}
if "camera_photos" not in st.session_state:
    st.session_state.camera_photos = {}
if "camera_videos" not in st.session_state:
    st.session_state.camera_videos = {}
    
# Track uploaded files for agent awareness
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

# --- Constants ---
WORKFLOW_ICONS = {
    "USER_QUERY": "👤",
    "LLM_THINKING": "☁️",
    "LLM_RESPONSE": "💬",
    "TOOL_CALL": "🔧",
    "TOOL_EXECUTION": "⚡️",
    "TOOL_RESULT": "📊",
    "FINAL_STATUS": "✅",
    "ERROR": "❌",
}

# Media handling functions
def extract_media_content(text):
    """
    Extract media content from text.
    Returns a tuple of (cleaned_text, media_items)
    where media_items is a list of dicts with 'type', 'content', and 'mime_type'
    """
    media_items = []
    cleaned_text = text
    
    # Regular expressions for different media patterns
    image_pattern = r"!\[.*?\]\((.*?)\)"  # Markdown image format ![alt](url)
    audio_pattern = r"<audio.*?src=[\"'](.*?)[\"'].*?>"  # HTML audio tag
    video_pattern = r"<video.*?src=[\"'](.*?)[\"'].*?>"  # HTML video tag
    
    # Base64 patterns
    base64_img_pattern = r"data:image/\w+;base64,([^\"'\s]+)"
    base64_audio_pattern = r"data:audio/\w+;base64,([^\"'\s]+)"
    base64_video_pattern = r"data:video/\w+;base64,([^\"'\s]+)"
    
    # File path patterns for Linux paths
    absolute_path_pattern = r"(/home/chengz/LAMs/mcp_chatbot-audio/output/(?:audio|video|plots)/[^:\*\?\"<>\|]+\.(mp3|wav|ogg|mp4|avi|mov|jpg|jpeg|png|gif))"
    relative_path_pattern = r"((?:audio|video|plots)/[^:\*\?\"<>\|]+\.(mp3|wav|ogg|mp4|avi|mov|jpg|jpeg|png|gif))"
    
    # Find markdown images
    for match in re.finditer(image_pattern, text):
        url = match.group(1)
        media_items.append({
            'type': 'image',
            'content': url,
            'mime_type': 'image/png'  # Assuming PNG as default
        })
        cleaned_text = cleaned_text.replace(match.group(0), "")
    
    # Find HTML audio tags
    for match in re.finditer(audio_pattern, text):
        url = match.group(1)
        media_items.append({
            'type': 'audio',
            'content': url,
            'mime_type': 'audio/mp3'  # Assuming MP3 as default
        })
        cleaned_text = cleaned_text.replace(match.group(0), "")
    
    # Find HTML video tags
    for match in re.finditer(video_pattern, text):
        url = match.group(1)
        media_items.append({
            'type': 'video',
            'content': url,
            'mime_type': 'video/mp4'  # Assuming MP4 as default
        })
        cleaned_text = cleaned_text.replace(match.group(0), "")
    
    # Find base64 encoded images
    for match in re.finditer(base64_img_pattern, text):
        b64_data = match.group(0)  # Full data URL
        media_items.append({
            'type': 'image',
            'content': b64_data,
            'mime_type': b64_data.split(';')[0].split(':')[1]
        })
        cleaned_text = cleaned_text.replace(match.group(0), "")
    
    # Find base64 encoded audio
    for match in re.finditer(base64_audio_pattern, text):
        b64_data = match.group(0)  # Full data URL
        media_items.append({
            'type': 'audio',
            'content': b64_data,
            'mime_type': b64_data.split(';')[0].split(':')[1]
        })
        cleaned_text = cleaned_text.replace(match.group(0), "")
    
    # Find base64 encoded video
    for match in re.finditer(base64_video_pattern, text):
        b64_data = match.group(0)  # Full data URL
        media_items.append({
            'type': 'video',
            'content': b64_data,
            'mime_type': b64_data.split(';')[0].split(':')[1]
        })
        cleaned_text = cleaned_text.replace(match.group(0), "")
    
    # Find absolute paths in output directory
    for match in re.finditer(absolute_path_pattern, text):
        file_path = match.group(1)
        ext = os.path.splitext(file_path)[1].lower()
        
        # Determine media type based on extension
        if ext in ['.mp3', '.wav', '.ogg']:
            media_type = 'audio'
            mime_type = f'audio/{ext[1:]}'
        elif ext in ['.mp4', '.avi', '.mov']:
            media_type = 'video'
            mime_type = f'video/{ext[1:]}'
        elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
            media_type = 'image'
            mime_type = f'image/{ext[1:].replace("jpg", "jpeg")}'
        else:
            continue  # Skip if not a recognized media file
            
        # Check if file exists
        if os.path.exists(file_path):
            media_items.append({
                'type': media_type,
                'content': file_path,
                'mime_type': mime_type,
                'is_file_path': True
            })
            # Don't remove the path from text as it's informative
    
    # Find relative paths in output directory
    for match in re.finditer(relative_path_pattern, text):
        rel_file_path = match.group(1)
        
        # Try multiple possible paths
        possible_paths = [
            os.path.join(OUTPUT_DIR, rel_file_path),  # Try with OUTPUT_DIR
            os.path.join(PROJECT_ROOT, rel_file_path),  # Try with PROJECT_ROOT
            rel_file_path,  # Try as-is
        ]
        
        for file_path in possible_paths:
            if os.path.exists(file_path):
                ext = os.path.splitext(file_path)[1].lower()
                
                # Determine media type based on extension
                if ext in ['.mp3', '.wav', '.ogg']:
                    media_type = 'audio'
                    mime_type = f'audio/{ext[1:]}'
                elif ext in ['.mp4', '.avi', '.mov']:
                    media_type = 'video'
                    mime_type = f'video/{ext[1:]}'
                elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
                    media_type = 'image'
                    mime_type = f'image/{ext[1:].replace("jpg", "jpeg")}'
                else:
                    continue  # Skip if not a recognized media file
                    
                media_items.append({
                    'type': media_type,
                    'content': file_path,
                    'mime_type': mime_type,
                    'is_file_path': True
                })
                break  # Stop once we find a valid path
            # Don't remove the path from text as it's informative
    
    return cleaned_text.strip(), media_items

def display_media(media_items):
    """Display media items using appropriate Streamlit components"""
    for item in media_items:
        try:
            if item.get('is_file_path', False):
                # This is a file path to media on disk
                file_path = item['content']
                
                if item['type'] == 'image':
                    try:
                        image = Image.open(file_path)
                        st.image(image, use_container_width=True, width=600)
                    except Exception as e:
                        st.error(f"Error displaying image from path: {e}")
                        st.markdown(f"Image path: `{file_path}`")
                
                elif item['type'] == 'audio':
                    try:
                        with open(file_path, 'rb') as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format=item['mime_type'].split('/')[1])
                    except Exception as e:
                        st.error(f"Error playing audio from path: {e}")
                        st.markdown(f"Audio path: `{file_path}`")
                
                elif item['type'] == 'video':
                    try:
                        with open(file_path, 'rb') as f:
                            video_bytes = f.read()
                        st.video(video_bytes, format=item['mime_type'].split('/')[1])
                    except Exception as e:
                        st.error(f"Error playing video from path: {e}")
                        st.markdown(f"Video path: `{file_path}`")
            
            # Handle normal media items (not file paths)
            elif item['type'] == 'image':
                if item['content'].startswith('data:image'):
                    # Handle base64 encoded image
                    content_type, content_string = item['content'].split(',')
                    decoded = base64.b64decode(content_string)
                    image = Image.open(BytesIO(decoded))
                    st.image(image, use_container_width=True, width=600)
                else:
                    # Handle URL
                    st.image(item['content'], use_container_width=True, width=600)
                    
            elif item['type'] == 'audio':
                if item['content'].startswith('data:audio'):
                    # Handle base64 encoded audio
                    content_type, content_string = item['content'].split(',')
                    decoded = base64.b64decode(content_string)
                    with BytesIO(decoded) as buffer:
                        st.audio(buffer, format=item['mime_type'].split('/')[1])
                else:
                    # Handle URL
                    st.audio(item['content'])
                    
            elif item['type'] == 'video':
                if item['content'].startswith('data:video'):
                    # Handle base64 encoded video
                    content_type, content_string = item['content'].split(',')
                    decoded = base64.b64decode(content_string)
                    with BytesIO(decoded) as buffer:
                        st.video(buffer, format=item['mime_type'].split('/')[1])
                else:
                    # Handle URL
                    st.video(item['content'])
        except Exception as e:
            st.error(f"Error displaying media: {e}")

def save_uploaded_file(uploaded_file, file_type):
    """Save uploaded file to disk and session state, and return a unique ID and file path"""
    file_name = uploaded_file.name
    
    # Check if file with same name already exists in the list to prevent duplicates
    for existing_file in st.session_state.uploaded_files_list:
        if existing_file['name'] == file_name and existing_file['type'] == file_type:
            # Return existing file id and path instead of creating duplicate
            return existing_file['id'], existing_file['path']
    
    file_id = str(uuid.uuid4())
    file_content = uploaded_file.getvalue()
    
    # Determine target directory and create full file path
    if file_type == 'image':
        target_dir = IMAGES_DIR
        file_dict = st.session_state.uploaded_images
    elif file_type == 'audio':
        target_dir = AUDIO_DIR
        file_dict = st.session_state.uploaded_audio
    elif file_type == 'video':
        target_dir = VIDEO_DIR
        file_dict = st.session_state.uploaded_videos
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # Create a filename with the original extension
    file_extension = os.path.splitext(file_name)[1]
    if not file_extension:
        # Default extensions if none is provided
        file_extension = {
            'image': '.png',
            'audio': '.mp3',
            'video': '.mp4'
        }.get(file_type, '')
    
    # Create a unique filename
    saved_filename = f"{file_type}_{file_id}{file_extension}"
    file_path = os.path.join(target_dir, saved_filename)
    
    # Save file to disk
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # Save info in session state
    file_info = {
        'content': file_content,
        'mime_type': uploaded_file.type,
        'name': file_name,
        'path': file_path,
        'relative_path': os.path.relpath(file_path, PROJECT_ROOT)
    }
    
    file_dict[file_id] = file_info
    
    # Add to uploaded files list for agent awareness
    st.session_state.uploaded_files_list.append({
        'id': file_id,
        'type': file_type,
        'name': file_name,
        'path': file_path,
        'relative_path': os.path.relpath(file_path, PROJECT_ROOT),
        'mime_type': uploaded_file.type
    })
    
    return file_id, file_path

# --- DataClass for Workflow Step ---
@dataclass
class WorkflowStep:
    """Workflow step class for tracking chatbot interactions."""

    type: str
    content: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


async def get_mcp_tools(force_refresh=False) -> Dict[str, List[MCPTool]]:
    """Get MCP tools from cache or by initializing clients."""
    if not force_refresh and st.session_state.mcp_tools_cache:
        return st.session_state.mcp_tools_cache

    tools_dict = {}
    config = st.session_state.chatbot_config
    server_config_path = os.path.join(
        PROJECT_ROOT, "mcp_servers", "servers_config.json"
    )
    if not os.path.exists(server_config_path):
        st.sidebar.warning("MCP Server config file not found. No tools loaded.")
        st.session_state.mcp_tools_cache = {}
        return {}

    try:
        server_config = config.load_config(server_config_path)
    except Exception as e:
        st.sidebar.error(f"Error loading MCP server config: {e}")
        st.session_state.mcp_tools_cache = {}
        return {}

    async with AsyncExitStack() as stack:
        if "mcpServers" not in server_config:
            st.sidebar.error(
                "Invalid MCP server config format: 'mcpServers' key missing."
            )
            st.session_state.mcp_tools_cache = {}
            return {}

        for name, srv_config in server_config["mcpServers"].items():
            try:
                client = MCPClient(name, srv_config)
                await stack.enter_async_context(client)
                tools = await client.list_tools()
                tools_dict[name] = tools
            except Exception as e:
                st.sidebar.error(f"Error fetching tools from {name}: {e}")

    st.session_state.mcp_tools_cache = tools_dict
    return tools_dict


def render_sidebar(mcp_tools: Optional[Dict[str, List[MCPTool]]] = None):
    """Render the sidebar with settings, MCP tools, and control buttons."""
    with st.sidebar:
        st.header("Settings")

        # --- Clear Chat Button ---
        if st.button("🧹 Clear Chat", use_container_width=True):
            # Clear chat history
            st.session_state.messages = []
            # Reset chat session state variables
            st.session_state.chat_session = None
            st.session_state.session_config_hash = None
            # Note: We don't explicitly close the AsyncExitStack here,
            # as it's difficult to do reliably from a synchronous button click
            # before rerun. The logic in process_chat handles cleanup when
            # a *new* session is created due to config change or None state.
            st.session_state.active_mcp_clients = []
            st.session_state.mcp_client_stack = None
            st.session_state.history_messages = []
            # Clear uploaded files list
            st.session_state.uploaded_files_list = []
            st.toast("Chat cleared!", icon="🧹")
            st.rerun()  # Rerun the app to reflect the cleared state

        # --- Media Upload Section ---
        with st.expander("📁 Upload Media", expanded=False):
            # Upload Image
            uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "gif"], key="image_uploader")
            if uploaded_image:
                file_id, file_path = save_uploaded_file(uploaded_image, 'image')
                st.image(uploaded_image, caption=uploaded_image.name, width=150)
                
                # 自动添加到上传文件列表
                rel_path = os.path.relpath(file_path, PROJECT_ROOT)
                # Check if file already exists in the list before adding it
                if not any(f['id'] == file_id for f in st.session_state.uploaded_files_list):
                    st.session_state.uploaded_files_list.append({
                        'id': file_id,
                        'type': 'image',
                        'name': uploaded_image.name,
                        'path': file_path,
                        'relative_path': rel_path,
                        'mime_type': uploaded_image.type
                    })
                    st.success(f"图片已上传: {uploaded_image.name}")
                else:
                    st.info(f"图片已存在: {uploaded_image.name}")
            
            # Audio upload
            uploaded_audio = st.file_uploader("Upload Audio", type=["mp3", "wav", "ogg"], key="audio_uploader")
            if uploaded_audio:
                file_id, file_path = save_uploaded_file(uploaded_audio, 'audio')
                st.audio(uploaded_audio)
                
                # 自动添加到上传文件列表
                rel_path = os.path.relpath(file_path, PROJECT_ROOT)
                # Check if file already exists in the list before adding it
                if not any(f['id'] == file_id for f in st.session_state.uploaded_files_list):
                    st.session_state.uploaded_files_list.append({
                        'id': file_id,
                        'type': 'audio',
                        'name': uploaded_audio.name,
                        'path': file_path,
                        'relative_path': rel_path,
                        'mime_type': uploaded_audio.type
                    })
                    st.success(f"音频已上传: {uploaded_audio.name}")
                else:
                    st.info(f"音频已存在: {uploaded_audio.name}")
            
            # Video upload
            uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"], key="video_uploader")
            if uploaded_video:
                file_id, file_path = save_uploaded_file(uploaded_video, 'video')
                st.video(uploaded_video)
                
                # 自动添加到上传文件列表
                rel_path = os.path.relpath(file_path, PROJECT_ROOT)
                # Check if file already exists in the list before adding it
                if not any(f['id'] == file_id for f in st.session_state.uploaded_files_list):
                    st.session_state.uploaded_files_list.append({
                        'id': file_id,
                        'type': 'video',
                        'name': uploaded_video.name,
                        'path': file_path,
                        'relative_path': rel_path,
                        'mime_type': uploaded_video.type
                    })
                    st.success(f"视频已上传: {uploaded_video.name}")
                else:
                    st.info(f"视频已存在: {uploaded_video.name}")
        
        # --- Media Recording Section ---
        with st.expander("🎙️ Record Media", expanded=False):
            st.info("🚧 功能尚未开放，请期待后续更新 🚧")
            st.write("媒体录制功能（录音、拍照、录像）即将上线...")

        llm_tab, mcp_tab = st.tabs(["LLM", "MCP"])
        with llm_tab:
            # LLM provider selection
            st.session_state.llm_provider = st.radio(
                "LLM Provider:",
                ["openai", "ollama"],
                index=["openai", "ollama"].index(st.session_state.llm_provider),
                key="llm_provider_radio",  # Add a key for stability
            )

            config = st.session_state.chatbot_config
            # Model selection based on provider
            if st.session_state.llm_provider == "openai":
                config._llm_model_name = st.text_input(
                    "OpenAI Model Name:",
                    value=config._llm_model_name or "gpt-3.5-turbo",
                    placeholder="e.g. gpt-4o",
                    key="openai_model_name",
                )
                config._llm_api_key = st.text_input(
                    "OpenAI API Key:",
                    value=config._llm_api_key or "",
                    type="password",
                    key="openai_api_key",
                )
                config._llm_base_url = st.text_input(
                    "OpenAI Base URL (optional):",
                    value=config._llm_base_url or "",
                    key="openai_base_url",
                )
            else:  # ollama
                config._ollama_model_name = st.text_input(
                    "Ollama Model Name:",
                    value=config._ollama_model_name or "llama3",
                    placeholder="e.g. llama3",
                    key="ollama_model_name",
                )
                config._ollama_base_url = st.text_input(
                    "Ollama Base URL:",
                    value=config._ollama_base_url or "http://localhost:11434",
                    key="ollama_base_url",
                )

        with mcp_tab:
            if st.button("🔄 Refresh Tools", use_container_width=True, type="primary"):
                st.session_state.mcp_tools_cache = {}
                # Also reset the session as tool changes might affect capabilities
                st.session_state.chat_session = None
                st.session_state.session_config_hash = None
                st.session_state.active_mcp_clients = []
                st.session_state.mcp_client_stack = None
                st.toast("Tools refreshed and session reset.", icon="🔄")
                st.rerun()

            if not mcp_tools:
                st.info("No MCP tools loaded or configured.")

            for client_name, client_tools in (mcp_tools or {}).items():
                with st.expander(f"Client: {client_name} ({len(client_tools)} tools)"):
                    if not client_tools:
                        st.write("No tools found for this client.")
                        continue
                    total_tools = len(client_tools)
                    for idx, tool in enumerate(client_tools):
                        st.markdown(f"**Tool {idx + 1}: `{tool.name}`**")
                        st.caption(f"{tool.description}")
                        # Use tool name in key for popover uniqueness
                        with st.popover("Schema"):
                            st.json(tool.input_schema)
                        if idx < total_tools - 1:
                            st.divider()

        # --- About Tabs ---
        en_about_tab, cn_about_tab = st.tabs(["About", "关于"])
        with en_about_tab:
            # st.markdown("### About") # Header inside tab might be too much
            st.info(
                "This chatbot uses the Model Context Protocol (MCP) for tool use. "
                "Configure LLM and MCP settings, then ask questions! "
                "Use the 'Clear Chat' button to reset the conversation."
            )
        with cn_about_tab:
            # st.markdown("### 关于")
            st.info(
                "这个聊天机器人使用模型上下文协议（MCP）进行工具使用。\n"
                "配置LLM和MCP设置，然后提出问题！使用 `Clear Chat` 按钮重置对话。"
            )


def extract_json_tool_calls(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """Extract tool call JSON objects from text using robust pattern matching.

    Uses similar logic to ChatSession._extract_tool_calls but adapted for our needs.

    Args:
        text: Text possibly containing JSON tool calls

    Returns:
        Tuple of (list of extracted tool calls, cleaned text without JSON)
    """
    tool_calls = []
    cleaned_text = text
    json_parsed = False

    # Try to parse the entire text as a single JSON array of
    # tool calls or a single tool call object
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):  # Check if it's a list of tool calls
            valid_tools = True
            for item in data:
                if not (
                    isinstance(item, dict) and "tool" in item and "arguments" in item
                ):
                    valid_tools = False
                    break
            if valid_tools:
                tool_calls.extend(data)
                json_parsed = True
        elif (
            isinstance(data, dict) and "tool" in data and "arguments" in data
        ):  # Check if it's a single tool call
            tool_calls.append(data)
            json_parsed = True

        if json_parsed:
            return (
                tool_calls,
                "",
            )  # Return empty string as cleaned text if parsing was successful

    except json.JSONDecodeError:
        pass  # Proceed to regex matching if direct parsing fails

    # Regex pattern to find potential JSON objects (might include tool calls)
    # This pattern tries to find JSON objects starting with '{' and ending with '}'
    json_pattern = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"
    matches = list(re.finditer(json_pattern, text))
    extracted_indices = set()

    for match in matches:
        start, end = match.span()
        # Avoid processing overlapping matches
        if any(
            start < prev_end and end > prev_start
            for prev_start, prev_end in extracted_indices
        ):
            continue

        json_str = match.group(0)
        try:
            obj = json.loads(json_str)
            # Check if the parsed object looks like a tool call
            if isinstance(obj, dict) and "tool" in obj and "arguments" in obj:
                tool_calls.append(obj)
                # Mark this region as extracted
                extracted_indices.add((start, end))
        except json.JSONDecodeError:
            # Ignore parts that are not valid JSON or not tool calls
            pass

    # Build the cleaned text by removing the extracted JSON parts
    if extracted_indices:
        cleaned_parts = []
        last_end = 0
        for start, end in sorted(list(extracted_indices)):
            cleaned_parts.append(text[last_end:start])
            last_end = end
        cleaned_parts.append(text[last_end:])
        cleaned_text = "".join(cleaned_parts).strip()
    else:
        # If no JSON tool calls were extracted via regex,
        # the original text is the cleaned text
        cleaned_text = text

    return tool_calls, cleaned_text


def render_workflow(steps: List[WorkflowStep], container=None):
    """Render workflow steps, placing each tool call sequence in its own expander."""
    if not steps:
        return

    target = container if container else st

    rendered_indices = set()

    # Iterate through steps to render them sequentially
    for i, step in enumerate(steps):
        if i in rendered_indices:
            continue

        step_type = step.type

        if step_type == "TOOL_CALL":
            # Start of a new tool call sequence
            tool_name = step.details.get("tool_name", "Unknown Tool")
            expander_title = f"{WORKFLOW_ICONS['TOOL_CALL']} Tool Call: {tool_name}"
            with target.expander(expander_title, expanded=False):
                # Display arguments
                arguments = step.details.get("arguments", {})
                st.write("**Arguments:**")
                if isinstance(arguments, str) and arguments == "Pending...":
                    st.write("Preparing arguments...")
                elif isinstance(arguments, dict):
                    if arguments:
                        for key, value in arguments.items():
                            st.write(f"- `{key}`: `{repr(value)}`")
                    else:
                        st.write("_No arguments_")
                else:
                    st.code(
                        str(arguments), language="json"
                    )  # Display as code block if not dict
                rendered_indices.add(i)

                # Look ahead for related execution and result steps for *this* tool call
                j = i + 1
                while j < len(steps):
                    next_step = steps[j]
                    # Associate based on sequence and type
                    if next_step.type == "TOOL_EXECUTION":
                        st.write(
                            f"**Status** {WORKFLOW_ICONS['TOOL_EXECUTION']}: "
                            f"{next_step.content}"
                        )
                        rendered_indices.add(j)
                    elif next_step.type == "TOOL_RESULT":
                        st.write(f"**Result** {WORKFLOW_ICONS['TOOL_RESULT']}:")
                        details = next_step.details
                        try:
                            # Success, tool execution completed.
                            details_dict = json.loads(details)
                            st.json(details_dict)
                        except json.JSONDecodeError:
                            # Error, tool execution failed.
                            result_str = str(details)
                            st.text(
                                result_str[:500]
                                + ("..." if len(result_str) > 500 else "")
                                or "_Empty result_"
                            )
                        rendered_indices.add(j)
                        break  # Stop looking ahead once result is found for this tool
                    elif (
                        next_step.type == "TOOL_CALL"
                        or next_step.type == "JSON_TOOL_CALL"
                    ):
                        # Stop if another tool call starts before finding the result
                        break
                    j += 1

        elif step_type == "JSON_TOOL_CALL":
            # Render LLM-generated tool calls in their own expander
            tool_name = step.details.get("tool_name", "Unknown")
            expander_title = (
                f"{WORKFLOW_ICONS['TOOL_CALL']} LLM Generated Tool Call: {tool_name}"
            )
            with target.expander(expander_title, expanded=False):
                st.write("**Arguments:**")
                arguments = step.details.get("arguments", {})
                if isinstance(arguments, dict):
                    if arguments:
                        for key, value in arguments.items():
                            st.write(f"- `{key}`: `{value}`")
                    else:
                        st.write("_No arguments_")
                else:
                    st.code(str(arguments), language="json")  # Display as code block
            rendered_indices.add(i)

        elif step_type == "ERROR":
            # Display errors directly, outside expanders
            target.error(f"{WORKFLOW_ICONS['ERROR']} {step.content}")
            rendered_indices.add(i)

        # Ignore other step types (USER_QUERY, LLM_THINKING, LLM_RESPONSE, FINAL_STATUS)
        # as they are handled elsewhere (status bar, main message area).


def get_config_hash(config: Configuration, provider: str) -> int:
    """Generate a hash based on relevant configuration settings."""
    relevant_config = {
        "provider": provider,
    }
    if provider == "openai":
        relevant_config.update(
            {
                "model": config._llm_model_name,
                "api_key": config._llm_api_key,
                "base_url": config._llm_base_url,
            }
        )
    else:  # ollama
        relevant_config.update(
            {
                "model": config._ollama_model_name,
                "base_url": config._ollama_base_url,
            }
        )
    # Hash the sorted representation for consistency
    return hash(json.dumps(relevant_config, sort_keys=True))


async def initialize_mcp_clients(
    config: Configuration, stack: AsyncExitStack
) -> List[MCPClient]:
    """Initializes MCP Clients based on config."""
    clients = []
    server_config_path = os.path.join(
        PROJECT_ROOT, "mcp_servers", "servers_config.json"
    )
    server_config = {}
    if os.path.exists(server_config_path):
        try:
            server_config = config.load_config(server_config_path)
        except Exception as e:
            st.warning(f"Failed to load MCP server config for client init: {e}")

    if server_config and "mcpServers" in server_config:
        for name, srv_config in server_config["mcpServers"].items():
            try:
                client = MCPClient(name, srv_config)
                # Enter the client's context into the provided stack
                await stack.enter_async_context(client)
                clients.append(client)
            except Exception as client_ex:
                st.error(f"Failed to initialize MCP client {name}: {client_ex}")
    return clients


async def process_chat(user_input: str):
    """Handles user input, interacts with the backend."""

    # 初始化关键变量，防止在错误处理路径中出现未定义错误
    media_items = []
    accumulated_response_content = ""
    chat_session = None
    clean_response = ""
    final_display_content = ""
    final_status_message = "Completed."
    
    # Check if the input is a media message (contains image/audio/video tags)
    contains_media = bool(re.search(r"data:(image|audio|video)", user_input))

    # 1. Add user message to state and display it
    # Use a copy for history to avoid potential modification issues if session resets
    if not st.session_state.messages or st.session_state.messages[-1]["role"] != "user" or st.session_state.messages[-1]["content"] != user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        if contains_media:
            # If message contains media, extract and display it properly
            cleaned_text, media_items = extract_media_content(user_input)
            if cleaned_text:
                st.markdown(cleaned_text)
            if media_items:
                display_media(media_items)
        else:
            # Regular text message
            st.markdown(user_input)

    # 2. Prepare for assistant response
    current_workflow_steps = []
    with st.chat_message("assistant"):
        status_placeholder = st.status("Processing...", expanded=False)
        workflow_display_container = st.empty()
        message_placeholder = st.empty()

    try:
        # Session and Client Management
        config = st.session_state.chatbot_config
        provider = st.session_state.llm_provider
        current_config_hash = get_config_hash(config, provider)

        # Check if config changed or session doesn't exist
        if (
            st.session_state.chat_session is None
            or current_config_hash != st.session_state.session_config_hash
        ):
            # If config changed, clear previous messages and reset state
            if (
                st.session_state.session_config_hash is not None
                and current_config_hash != st.session_state.session_config_hash
            ):
                st.session_state.messages = [
                    {"role": "user", "content": user_input}
                ]  # Keep only current input
                # Need to properly exit previous stack if it exists
                if st.session_state.mcp_client_stack:
                    await st.session_state.mcp_client_stack.__aexit__(None, None, None)
                    st.session_state.mcp_client_stack = None
                    st.session_state.active_mcp_clients = []

            # Create LLM Client
            llm_client = create_llm_client(provider=provider, config=config)
            if not llm_client:
                raise ValueError(
                    "LLM Client could not be created. Check configuration."
                )

            # Create and manage MCP Clients using
            # an AsyncExitStack stored in session state
            st.session_state.mcp_client_stack = AsyncExitStack()
            mcp_clients = await initialize_mcp_clients(
                config, st.session_state.mcp_client_stack
            )
            st.session_state.active_mcp_clients = mcp_clients  # Store references

            # Create new ChatSession
            # Pass the *active* clients.
            # ChatSession needs to handle these potentially changing.
            # Assuming ChatSession uses the clients passed at creation time.
            st.session_state.chat_session = ChatSession(
                st.session_state.active_mcp_clients, llm_client
            )
            await st.session_state.chat_session.initialize()
            # Keep the history messages from the new chat session.
            if not st.session_state.history_messages:
                # If the history messages are not set, we need to get the
                # system prompt from the chat session.
                st.session_state.history_messages = (
                    st.session_state.chat_session.messages
                )
            st.session_state.session_config_hash = current_config_hash
        else:
            # Ensure clients are available if session exists
            # (they should be in active_mcp_clients)
            # This part assumes the clients associated with the existing session
            # are still valid. If tool refresh happens, this might need adjustment.
            pass  # Use existing session

        if not st.session_state.chat_session:
            raise RuntimeError("Chat session could not be initialized.")

        chat_session = st.session_state.chat_session
        chat_session.messages = st.session_state.history_messages
        print("Chat session messages:", chat_session.messages)
        
        # Add information about uploaded files to user's message if there are any
        if st.session_state.uploaded_files_list:
            files_info = "已上传的多媒体文件列表：\n"
            for file in st.session_state.uploaded_files_list:
                files_info += f"- {file['name']} ({file['type']}): {file['relative_path']}\n"
            
            # 修改逻辑：始终提供文件信息，并根据用户问题调整优先级
            is_file_query = any(word in user_input.lower() for word in ["文件", "上传", "file", "upload", "图片", "音频", "视频", "image", "audio", "video", "什么", "哪些", "list"])
            
            if is_file_query:
                # 用户询问文件，将文件列表放在前面并明确标注
                enhanced_input = f"用户询问了上传的文件信息。\n\n{files_info}\n\n用户原始问题: {user_input}"
            else:
                # 其他问题，将文件信息放在后面作为上下文
                enhanced_input = f"{user_input}\n\n系统信息 - 当前已上传的文件：\n{files_info}"
            
            print(f"Enhanced user input with files info: {enhanced_input}")
        else:
            # 没有上传文件时，直接使用原始输入
            enhanced_input = user_input

        # Add user query to workflow steps
        current_workflow_steps.append(
            WorkflowStep(type="USER_QUERY", content=enhanced_input)
        )
        with workflow_display_container.container():
            render_workflow([], container=st)  # Render empty initially

        tool_call_count = 0
        active_tool_name = None
        mcp_tool_calls_made = False

        # Initial thinking step (not added to workflow steps for display here)
        status_placeholder.update(
            label="🧠 Processing request...", state="running", expanded=False
        )

        # Stream response handling
        accumulated_response_content = ""  # Accumulate raw response content
        new_step_added = False  # Track if workflow needs rerender

        # Process streaming response using the persistent chat_session
        print("Now chat session messages:", chat_session.messages)
        async for result in chat_session.send_message_stream(
            enhanced_input, show_workflow=True
        ):
            new_step_added = False  # Reset for this iteration
            if isinstance(result, tuple):
                status, content = result

                if status == "status":
                    status_placeholder.update(label=f"🧠 {content}", state="running")
                elif status == "tool_call":
                    mcp_tool_calls_made = True
                    tool_call_count += 1
                    active_tool_name = content
                    tool_call_step = WorkflowStep(
                        type="TOOL_CALL",
                        content=f"Initiating call to: {content}",
                        details={"tool_name": content, "arguments": "Pending..."},
                    )
                    current_workflow_steps.append(tool_call_step)
                    new_step_added = True
                    status_placeholder.update(
                        label=f"🔧 Calling tool: {content}", state="running"
                    )
                elif status == "tool_arguments":
                    if active_tool_name:
                        updated = False
                        for step in reversed(current_workflow_steps):
                            if (
                                step.type == "TOOL_CALL"
                                and step.details.get("tool_name") == active_tool_name
                                and step.details.get("arguments") == "Pending..."
                            ):
                                try:
                                    step.details["arguments"] = json.loads(content)
                                except json.JSONDecodeError:
                                    step.details["arguments"] = content
                                updated = True
                                break
                        if updated:
                            new_step_added = True
                elif status == "tool_execution":
                    current_workflow_steps.append(
                        WorkflowStep(type="TOOL_EXECUTION", content=content)
                    )
                    new_step_added = True
                    status_placeholder.update(label=f"⚡ {content}", state="running")
                elif status == "tool_results":
                    current_workflow_steps.append(
                        WorkflowStep(
                            type="TOOL_RESULT",
                            content="Received result.",
                            details=content,
                        )
                    )
                    new_step_added = True
                    status_placeholder.update(
                        label=f"🧠 Processing results from {active_tool_name}...",
                        state="running",
                    )
                    active_tool_name = None

                elif status == "response":
                    if isinstance(content, str):
                        accumulated_response_content += content
                        potential_json_tools, clean_response_so_far = (
                            extract_json_tool_calls(accumulated_response_content)
                        )
                        
                        # Check for media content in streaming response
                        cleaned_text, media_items = extract_media_content(clean_response_so_far)
                        
                        # Display text without media (we'll display media at the end)
                        message_placeholder.markdown(cleaned_text + "▌")
                        status_placeholder.update(
                            label="💬 Streaming response...", state="running"
                        )

                elif status == "error":
                    error_content = str(content)
                    error_step = WorkflowStep(type="ERROR", content=error_content)
                    current_workflow_steps.append(error_step)
                    new_step_added = True
                    status_placeholder.update(
                        label=f"❌ Error: {error_content[:100]}...",
                        state="error",
                        expanded=True,
                    )
                    message_placeholder.error(f"An error occurred: {error_content}")
                    with workflow_display_container.container():
                        render_workflow(current_workflow_steps, container=st)
                    break  # Stop processing on error

            else:  # Handle non-tuple results (e.g., direct string) if necessary
                if isinstance(result, str):
                    accumulated_response_content += result
                    potential_json_tools, clean_response_so_far = (
                        extract_json_tool_calls(accumulated_response_content)
                    )
                    
                    # Handle media in streaming response
                    cleaned_text, media_items = extract_media_content(clean_response_so_far)
                    message_placeholder.markdown(cleaned_text + "▌")
                    status_placeholder.update(
                        label="💬 Streaming response...", state="running"
                    )

            # Re-render the workflow area if a new step was added
            if new_step_added:
                with workflow_display_container.container():
                    render_workflow(current_workflow_steps, container=st)

        # 3. Post-stream processing and final display

        json_tools, clean_response = extract_json_tool_calls(
            accumulated_response_content
        )
        
        # Extract media content from the final response
        final_text, media_items = extract_media_content(clean_response.strip())
        final_display_content = final_text

        json_tools_added = False
        for json_tool in json_tools:
            if not mcp_tool_calls_made:  # Heuristic: only add if no standard calls
                tool_name = json_tool.get("tool", "unknown_tool")
                tool_args = json_tool.get("arguments", {})
                json_step = WorkflowStep(
                    type="JSON_TOOL_CALL",
                    content=f"LLM generated tool call: {tool_name}",
                    details={"tool_name": tool_name, "arguments": tool_args},
                )
                current_workflow_steps.append(json_step)
                tool_call_count += 1
                json_tools_added = True

        if not final_display_content and json_tools_added:
            final_display_content = ""  # Or a message like "Generated tool calls."

        # Clear the placeholder before adding new content
        message_placeholder.empty()
        
        # Create a container for both text and media
        with message_placeholder.container():
            # Display the final text content
            if final_display_content:
                st.markdown(final_display_content)
            
            # Display any media items if present
            if media_items:
                # Remove any duplicate media items based on content
                unique_media_items = []
                seen_contents = set()
                
                for item in media_items:
                    # For file paths, use the path as identifier
                    if item.get('is_file_path', False):
                        content_id = item['content']
                    else:
                        # For base64 or URLs, use first 100 chars as identifier
                        content_id = str(item['content'])[:100]
                    
                    if content_id not in seen_contents:
                        seen_contents.add(content_id)
                        unique_media_items.append(item)
                
                # Display the unique media items
                display_media(unique_media_items)

        if final_display_content or media_items:
            llm_response_step = WorkflowStep(
                type="LLM_RESPONSE",
                content="Final response generated.",
                details={"response_text": final_display_content, "has_media": bool(media_items)},
            )
            current_workflow_steps.append(llm_response_step)

        final_status_message = "Completed."
        if tool_call_count > 0:
            final_status_message += f" Processed {tool_call_count} tool call(s)."
        if media_items:
            final_status_message += f" Generated {len(media_items)} media item(s)."
        
        current_workflow_steps.append(
            WorkflowStep(type="FINAL_STATUS", content=final_status_message)
        )

        status_placeholder.update(
            label=f"✅ {final_status_message}", state="complete", expanded=False
        )

        with workflow_display_container.container():
            render_workflow(current_workflow_steps, container=st)

        # --- Store results in session state ---
        # Find the last user message added
        last_user_message_index = -1
        for i in range(len(st.session_state.messages) - 1, -1, -1):
            if st.session_state.messages[i]["role"] == "user":
                last_user_message_index = i
                break

        # Append assistant message right after the last user message
        assistant_message = {
            "role": "assistant",
            "content": clean_response.strip() or accumulated_response_content,  # Store with media
            "workflow_steps": [step.to_dict() for step in current_workflow_steps],
        }
        if last_user_message_index != -1:
            st.session_state.messages.insert(
                last_user_message_index + 1, assistant_message
            )
        else:
            # Should not happen if we added user message first, but as fallback
            st.session_state.messages.append(assistant_message)
        # --- End storing results ---

    except Exception as e:
        error_message = f"An unexpected error occurred in process_chat: {str(e)}"
        st.error(error_message)
        current_workflow_steps.append(WorkflowStep(type="ERROR", content=error_message))
        try:
            with workflow_display_container.container():
                render_workflow(current_workflow_steps, container=st)
        except Exception as render_e:
            st.error(f"Additionally, failed to render workflow after error: {render_e}")

        status_placeholder.update(
            label=f"❌ Error: {error_message[:100]}...", state="error", expanded=True
        )
        # Append error message to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Error: {error_message}",
                "workflow_steps": [step.to_dict() for step in current_workflow_steps],
            }
        )
    finally:
        # --- Final UI update ---
        if (
            status_placeholder._label != f"✅ {final_status_message}"
            and status_placeholder._state != "error"
        ):
            status_placeholder.update(
                label="Processing finished.", state="complete", expanded=False
            )

        # ------------------------------------------------------------------
        # IMPORTANT CLEAN‑UP!
        #
        # Each Streamlit rerun executes this script in a *fresh* asyncio
        # event‑loop.  Any MCPClient / ChatSession objects created in a
        # previous loop become invalid and will raise
        # "Attempted to exit cancel scope in a different task…" errors when
        # they try to close themselves later on.
        #
        # Therefore we:
        #   1. Close the AsyncExitStack that owns all MCP clients *inside the
        #      same loop that created them* (`process_chat`'s loop).
        #   2. Drop the references from `st.session_state` so a new set of
        #      clients / ChatSession are created on the next user message.
        # ------------------------------------------------------------------
        try:
            if st.session_state.mcp_client_stack is not None:
                await st.session_state.mcp_client_stack.__aexit__(None, None, None)
        except Exception as cleanup_exc:
            # Log but do not crash UI – the loop is ending anyway.
            print("MCP clean‑up error:", cleanup_exc, file=sys.stderr)
        finally:
            st.session_state.mcp_client_stack = None
            st.session_state.active_mcp_clients = []
            # Do *not* reuse async objects across Streamlit reruns.
            st.session_state.history_messages = chat_session.messages
            st.session_state.chat_session = None


def display_chat_history():
    """Displays the chat history from st.session_state.messages."""
    for idx, message in enumerate(st.session_state.messages):
        # Use a unique key for each chat message element
        with st.chat_message(message["role"]):
            # Workflow Rendering
            if message["role"] == "assistant" and "workflow_steps" in message:
                # Use a unique key for the workflow container
                workflow_history_container = st.container()

                workflow_steps = []
                if isinstance(message["workflow_steps"], list):
                    for step_dict in message["workflow_steps"]:
                        if isinstance(step_dict, dict):
                            workflow_steps.append(
                                WorkflowStep(
                                    type=step_dict.get("type", "UNKNOWN"),
                                    content=step_dict.get("content", ""),
                                    details=step_dict.get("details", {}),
                                )
                            )
                if workflow_steps:
                    render_workflow(
                        workflow_steps, container=workflow_history_container
                    )

            # Message Content Rendering (Rendered after workflow for assistant)
            content = message["content"]
            
            # Check for media content in the message
            cleaned_content, media_items = extract_media_content(content)
            
            # Create container for content to keep text and media together
            with st.container():
                # Display text content if any remains after extracting media
                if cleaned_content:
                    st.markdown(cleaned_content, unsafe_allow_html=True)
                
                # Remove any duplicate media items based on content
                if media_items:
                    unique_media_items = []
                    seen_contents = set()
                    
                    for item in media_items:
                        # For file paths, use the path as identifier
                        if item.get('is_file_path', False):
                            content_id = item['content']
                        else:
                            # For base64 or URLs, use first 100 chars as identifier
                            content_id = str(item['content'])[:100]
                        
                        if content_id not in seen_contents:
                            seen_contents.add(content_id)
                            unique_media_items.append(item)
                    
                    # Display only unique media items
                    display_media(unique_media_items)


async def main():
    """Main application entry point."""
    # Get MCP tools (cached) - Tool list displayed in sidebar
    mcp_tools = await get_mcp_tools()

    # Render sidebar - Allows config changes and clearing chat
    render_sidebar(mcp_tools)

    # Display existing chat messages and their workflows from session state
    display_chat_history()

    # Handle new chat input
    if prompt := st.chat_input(
        "Ask something... (e.g., 'What files are in the root directory?')"
    ):
        await process_chat(prompt)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Note: Reliable async cleanup on shutdown is still complex in Streamlit
        pass
