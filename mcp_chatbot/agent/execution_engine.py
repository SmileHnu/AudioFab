"""
Execution Engine for AudioFab Agent System

Optimized execution engine with caching, parallel processing, and error handling.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from ..mcp.client import MCPClient

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Optimized execution engine for AudioFab tools."""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.execution_cache = {}
        self.active_executions = {}
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single tool with full optimization.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            
        Returns:
            Execution result with metadata
        """
        start_time = time.time()
        execution_id = f"{tool_name}_{int(start_time * 1000)}"
        
        try:
            # Check if already executing
            if execution_id in self.active_executions:
                return self.active_executions[execution_id]
            
            # Mark as active
            self.active_executions[execution_id] = {"status": "running"}
            
            # Validate parameters
            validation_result = await self._validate_parameters(tool_name, parameters)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": f"Parameter validation failed: {validation_result['error']}"
                }
            
            # Execute tool
            result = await self._execute_mcp_tool(tool_name, parameters)
            
            # Process result
            processed_result = await self._process_result(tool_name, result)
            
            # Add metadata
            processed_result["execution_metadata"] = {
                "tool_name": tool_name,
                "execution_id": execution_id,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": time.time() - start_time,
                "parameters": parameters
            }
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name,
                "parameters": parameters
            }
        finally:
            # Clean up active execution
            self.active_executions.pop(execution_id, None)
    
    async def execute_parallel(self, tool_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute multiple tools in parallel.
        
        Args:
            tool_tasks: List of {"tool": str, "parameters": dict}
            
        Returns:
            Combined results
        """
        if not tool_tasks:
            return {"success": True, "results": {}}
        
        # Group tasks by compatibility
        compatible_groups = self._group_compatible_tasks(tool_tasks)
        
        all_results = {}
        
        for group in compatible_groups:
            if len(group) == 1:
                # Sequential execution
                task = group[0]
                result = await self.execute_tool(task["tool"], task["parameters"])
                all_results[task["tool"]] = result
            else:
                # Parallel execution
                parallel_results = await self._execute_parallel_group(group)
                all_results.update(parallel_results)
        
        return {
            "success": all(r.get("success", False) for r in all_results.values()),
            "results": all_results,
            "parallel_groups": len(compatible_groups)
        }
    
    async def _execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool via MCP client."""
        try:
            # Call tool via MCP
            result = await self.mcp_client.call_tool(tool_name, parameters)
            
            # Parse result
            if isinstance(result, dict):
                return result
            elif hasattr(result, 'content'):
                # Handle MCP result objects
                content = result.content
                if isinstance(content, dict):
                    return content
                elif isinstance(content, str):
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return {"success": True, "result": content}
                else:
                    return {"success": True, "result": str(content)}
            else:
                return {"success": True, "result": str(result)}
                
        except Exception as e:
            logger.error(f"MCP tool execution error for {tool_name}: {e}")
            return {
                "success": False,
                "error": f"MCP execution failed: {str(e)}"
            }
    
    async def _validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool parameters."""
        try:
            # Basic validation
            if not isinstance(parameters, dict):
                return {"valid": False, "error": "Parameters must be a dictionary"}
            
            # Tool-specific validation can be added here
            # For now, basic structure validation
            required_params = self._get_required_params(tool_name)
            
            missing_params = []
            for param in required_params:
                if param not in parameters:
                    missing_params.append(param)
            
            if missing_params:
                return {
                    "valid": False,
                    "error": f"Missing required parameters: {missing_params}"
                }
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def _get_required_params(self, tool_name: str) -> List[str]:
        """Get required parameters for a tool."""
        # This would ideally come from tool schema
        # For now, basic parameter mapping
        param_mapping = {
            "MusicGenTool": ["prompt"],
            "AudioGenTool": ["prompt"],
            "CosyVoice2Tool": ["text"],
            "SparkTTSTool": ["text"],
            "FunASRTool": ["audio_path"],
            "WhisperASRTool": ["audio_path"],
            "ClearVoiceTool": ["audio_path"],
            "DiffRhythmTool": ["lrc_text"],
            "ACEStepTool": ["prompt"],
            "VoiceCraftTool": ["audio_path", "edit_text"],
            "AudioXTool": ["prompt"],
            "Hallo2Tool": ["image_path", "audio_path"],
            "TIGERSpeechSeparationTool": ["audio_path"],
            "AudioSeparatorTool": ["audio_path"]
        }
        
        return param_mapping.get(tool_name, [])
    
    def _group_compatible_tasks(self, tool_tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group tasks for optimal parallel execution."""
        # Simple grouping - can be enhanced with dependency analysis
        memory_intensive = ["Hallo2Tool", "AudioXTool", "DiffRhythmTool", "ACEStepTool"]
        cpu_intensive = ["TIGERSpeechSeparationTool", "AudioSeparatorTool", "ClearVoiceTool"]
        
        groups = []
        current_group = []
        
        for task in tool_tasks:
            tool = task["tool"]
            
            # Simple heuristic: avoid mixing memory-intensive tools
            if tool in memory_intensive:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([task])
            else:
                current_group.append(task)
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    async def _execute_parallel_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a group of tasks in parallel."""
        async def execute_single(task):
            result = await self.execute_tool(task["tool"], task["parameters"])
            return task["tool"], result
        
        tasks = [execute_single(task) for task in group]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = {}
        for tool_name, result in results:
            if isinstance(result, Exception):
                final_results[tool_name] = {
                    "success": False,
                    "error": str(result)
                }
            else:
                final_results[tool_name] = result
        
        return final_results
    
    async def _process_result(self, tool_name: str, result: Any) -> Dict[str, Any]:
        """Process and enrich tool execution results."""
        if isinstance(result, dict):
            processed = dict(result)
        else:
            processed = {"result": result}
        
        # Add enrichment based on tool type
        enrichment = await self._enrich_result(tool_name, processed)
        processed.update(enrichment)
        
        return processed
    
    async def _enrich_result(self, tool_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich results with additional metadata."""
        enrichment = {}
        
        # Add file metadata if output path exists
        if "output_path" in result:
            output_path = Path(result["output_path"])
            if output_path.exists():
                enrichment.update({
                    "file_size": output_path.stat().st_size,
                    "file_extension": output_path.suffix,
                    "creation_time": datetime.fromtimestamp(output_path.stat().st_ctime).isoformat()
                })
        
        # Tool-specific enrichments
        if tool_name == "MusicGenTool" and "prompt" in result:
            enrichment["genre_analysis"] = await self._analyze_genre(result.get("prompt", ""))
        
        if tool_name == "FunASRTool" and "result" in result:
            enrichment["text_analysis"] = await self._analyze_text(result["result"])
        
        return enrichment
    
    async def _analyze_genre(self, prompt: str) -> str:
        """Analyze music genre from prompt."""
        genre_keywords = {
            "jazz": ["jazz", "blues", "swing", "improvisation"],
            "rock": ["rock", "guitar", "drums", "band"],
            "classical": ["classical", "orchestra", "symphony", "piano"],
            "electronic": ["electronic", "synth", "edm", "techno"],
            "pop": ["pop", "catchy", "melody", "chorus"],
            "hiphop": ["hiphop", "rap", "beat", "rhythm"]
        }
        
        prompt_lower = prompt.lower()
        for genre, keywords in genre_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return genre
        
        return "general"
    
    async def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze transcribed text."""
        return {
            "word_count": len(text.split()),
            "character_count": len(text),
            "estimated_duration": len(text.split()) * 0.6  # Rough estimate
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "total_executions": len(self.execution_cache),
            "active_executions": len(self.active_executions),
            "cache_size": len(self.execution_cache),
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_cache(self):
        """Clear execution cache."""
        self.execution_cache.clear()
        self.active_executions.clear()
        logger.info("Execution cache cleared")