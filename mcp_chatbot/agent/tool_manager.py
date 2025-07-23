"""
Tool Manager for AudioFab Agent System

Manages tool discovery, selection, and optimization strategies.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


class ToolManager:
    """Manages tool discovery, selection, and optimization."""
    
    def __init__(self):
        self.available_tools = {}
        self.tool_categories = {}
        self.tool_dependencies = {}
        self.optimization_hints = {}
    
    async def load_tools(self, mcp_client):
        """Load available tools from MCP client."""
        try:
            # Get tools from MCP client
            tools_response = await mcp_client.list_tools()
            
            # Filter for local tools only (ignore API-based tools)
            self.available_tools = self._filter_local_tools(tools_response)
            
            # Categorize tools
            self.tool_categories = self._categorize_tools(self.available_tools)
            
            # Build dependency map
            self.tool_dependencies = self._build_dependency_map()
            
            # Generate optimization hints
            self.optimization_hints = self._generate_optimization_hints()
            
            logger.info(f"Loaded {len(self.available_tools)} local tools")
            
        except Exception as e:
            logger.error(f"Failed to load tools: {e}")
            # Fallback to local tool loading
            await self._load_local_tools_fallback()
    
    def _filter_local_tools(self, tools_response) -> Dict[str, Any]:
        """Filter out API-based tools, keep only local tools."""
        local_tools = {}
        
        # Load tool descriptions from JSON
        tool_desc_path = Path("mcp_servers/tool_descriptions.json")
        if tool_desc_path.exists():
            with open(tool_desc_path, 'r', encoding='utf-8') as f:
                all_tools = json.load(f)
                
                # Filter for local tools based on patterns
                for tool_name, tool_info in all_tools.items():
                    if self._is_local_tool(tool_name, tool_info):
                        local_tools[tool_name] = tool_info
        
        return local_tools
    
    def _is_local_tool(self, tool_name: str, tool_info: Dict[str, Any]) -> bool:
        """Determine if a tool is locally executable."""
        # Exclude API-based tools
        api_tools = [
            "API_servers", "external_api", "huggingface", "modelscope",
            "online", "cloud", "remote"
        ]
        
        description = tool_info.get("description", "").lower()
        name_lower = tool_name.lower()
        
        # Check for API indicators
        for api_indicator in api_tools:
            if api_indicator in description or api_indicator in name_lower:
                return False
        
        # Include local processing tools
        local_indicators = [
            "processor", "local", "generate", "synthesis", "recognition",
            "separation", "enhancement", "editing", "analysis"
        ]
        
        for local_indicator in local_indicators:
            if local_indicator in description or local_indicator in name_lower:
                return True
        
        # Default inclusion for tools with processors
        processor_files = [
            "ACE_step_processor.py", "AudioX_processor.py", "Audiocraft_tool_processor.py",
            "Audiosep_processor.py", "ClearerVoice_processor.py", "Cosyvoice2_tool.py",
            "DiffRhythm_processor.py", "Funasr_processor.py", "Qwen2Audio_processor.py",
            "Sparktts_processor.py", "TIGER_speech_separation_processor.py",
            "VoiceCraft_processor.py", "audio_separator_processor.py", "audiosr_tool.py",
            "hello2_processor.py", "openai_audio_processor.py", "whisper_tool.py",
            "yue_e_tool.py"
        ]
        
        tool_base_name = tool_name.replace("Tool", "").lower()
        for processor_file in processor_files:
            if tool_base_name in processor_file.lower():
                return True
        
        return False
    
    def _categorize_tools(self, tools: Dict[str, Any]) -> Dict[str, List[str]]:
        """Categorize tools by functionality."""
        categories = {
            "speech": [],
            "music": [],
            "enhancement": [],
            "analysis": [],
            "generation": [],
            "editing": [],
            "separation": []
        }
        
        for tool_name, tool_info in tools.items():
            description = tool_info.get("description", "").lower()
            name_lower = tool_name.lower()
            
            # Speech tools
            if any(word in description or name_lower for word in ["speech", "voice", "asr", "tts"]):
                categories["speech"].append(tool_name)
            
            # Music tools
            if any(word in description or name_lower for word in ["music", "song", "melody", "rhythm"]):
                categories["music"].append(tool_name)
            
            # Enhancement tools
            if any(word in description or name_lower for word in ["enhance", "improve", "clean", "noise"]):
                categories["enhancement"].append(tool_name)
            
            # Analysis tools
            if any(word in description or name_lower for word in ["analysis", "recognition", "detect", "identify"]):
                categories["analysis"].append(tool_name)
            
            # Generation tools
            if any(word in description or name_lower for word in ["generate", "create", "synthesis", "produce"]):
                categories["generation"].append(tool_name)
            
            # Editing tools
            if any(word in description or name_lower for word in ["edit", "modify", "change", "replace"]):
                categories["editing"].append(tool_name)
            
            # Separation tools
            if any(word in description or name_lower for word in ["separate", "split", "isolate", "extract"]):
                categories["separation"].append(tool_name)
        
        return categories
    
    def _build_dependency_map(self) -> Dict[str, List[str]]:
        """Build dependency relationships between tools."""
        dependencies = {
            "VoiceCraftTool": ["FunASRTool"],  # May need transcription
            "MusicGenTool": [],  # Independent
            "AudioGenTool": [],  # Independent
            "ClearVoiceTool": [],  # Can work independently
            "DiffRhythmTool": [],  # Independent
            "ACEStepTool": [],  # Independent
            "CosyVoice2Tool": [],  # Independent
            "SparkTTSTool": [],  # Independent
            "AudioXTool": [],  # Independent
            "Hallo2Tool": ["FunASRTool"],  # May need audio analysis
            "FunASRTool": [],  # Independent
            "Qwen2AudioTool": [],  # Independent
            "WhisperASRTool": [],  # Independent
            "TIGERSpeechSeparationTool": [],  # Independent
            "AudioSeparatorTool": []  # Independent
        }
        
        return dependencies
    
    def _generate_optimization_hints(self) -> Dict[str, Any]:
        """Generate optimization hints for tool usage."""
        return {
            "parallel_safe": [
                "MusicGenTool", "AudioGenTool", "CosyVoice2Tool", "SparkTTSTool",
                "ACEStepTool", "DiffRhythmTool"
            ],
            "memory_intensive": [
                "Hallo2Tool", "AudioXTool", "DiffRhythmTool", "ACEStepTool"
            ],
            "cpu_intensive": [
                "TIGERSpeechSeparationTool", "AudioSeparatorTool", "ClearVoiceTool"
            ],
            "fast_tools": [
                "FunASRTool", "WhisperASRTool", "Qwen2AudioTool"
            ],
            "gpu_preferred": [
                "MusicGenTool", "AudioGenTool", "Hallo2Tool", "AudioXTool",
                "DiffRhythmTool", "ACEStepTool"
            ]
        }
    
    async def _load_local_tools_fallback(self):
        """Fallback method to load local tools from JSON."""
        try:
            tool_desc_path = Path("mcp_servers/tool_descriptions.json")
            if tool_desc_path.exists():
                with open(tool_desc_path, 'r', encoding='utf-8') as f:
                    all_tools = json.load(f)
                    
                # Filter for local tools
                self.available_tools = {
                    name: info for name, info in all_tools.items()
                    if self._is_local_tool(name, info)
                }
                
                logger.info(f"Loaded {len(self.available_tools)} tools from fallback")
        except Exception as e:
            logger.error(f"Fallback tool loading failed: {e}")
    
    def get_tools_summary(self) -> Dict[str, Any]:
        """Get summary of available tools."""
        return {
            "total_tools": len(self.available_tools),
            "categories": self.tool_categories,
            "tools": {
                name: {
                    "description": info.get("description", ""),
                    "category": self._get_tool_category(name),
                    "parameters": list(info.get("parameters", {}).keys())
                }
                for name, info in self.available_tools.items()
            }
        }
    
    def _get_tool_category(self, tool_name: str) -> str:
        """Get primary category for a tool."""
        for category, tools in self.tool_categories.items():
            if tool_name in tools:
                return category
        return "general"
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.available_tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tool."""
        return self.available_tools.get(tool_name)
    
    def get_compatible_tools(self, task_type: str) -> List[str]:
        """Get tools compatible with a specific task type."""
        return self.tool_categories.get(task_type, [])
    
    def get_optimization_hints(self, tool_name: str) -> Dict[str, Any]:
        """Get optimization hints for a specific tool."""
        hints = {}
        
        for hint_type, tool_list in self.optimization_hints.items():
            hints[hint_type] = tool_name in tool_list
        
        return hints
    
    def get_execution_order(self, tools: List[str]) -> List[str]:
        """Get optimal execution order considering dependencies."""
        ordered_tools = []
        visited = set()
        
        def add_dependencies(tool):
            if tool in visited:
                return
            
            # Add dependencies first
            for dep in self.tool_dependencies.get(tool, []):
                if dep in tools and dep not in visited:
                    add_dependencies(dep)
            
            ordered_tools.append(tool)
            visited.add(tool)
        
        for tool in tools:
            if tool not in visited:
                add_dependencies(tool)
        
        return ordered_tools