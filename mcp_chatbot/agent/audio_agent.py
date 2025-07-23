"""
AudioFab High-Performance Agent System

Core agent implementation with intelligent tool selection,
caching, and optimized execution strategies.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

from .tool_manager import ToolManager
from .execution_engine import ExecutionEngine
from .cache_manager import CacheManager
from ..llm.oai import OpenAIClient
from ..mcp.client import MCPClient

logger = logging.getLogger(__name__)


class AudioFabAgent:
    """
    High-performance audio processing agent with intelligent capabilities.
    
    Features:
    - Smart tool selection based on context and history
    - Intelligent caching for repeated operations
    - Parallel execution when possible
    - Error recovery and retry strategies
    - Performance optimization and monitoring
    """
    
    def __init__(self, mcp_client: MCPClient, llm_client: OpenAIClient):
        self.mcp_client = mcp_client
        self.llm_client = llm_client
        
        # Core components
        self.tool_manager = ToolManager()
        self.execution_engine = ExecutionEngine(mcp_client)
        self.cache_manager = CacheManager()
        
        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "tool_executions": 0,
            "average_response_time": 0.0,
            "error_rate": 0.0
        }
        
        # Configuration
        self.max_parallel_tasks = 3
        self.cache_ttl_hours = 24
        self.retry_attempts = 3
        self.retry_delay = 1.0
        
    async def initialize(self):
        """Initialize the agent system."""
        logger.info("Initializing AudioFab Agent System...")
        
        # Initialize MCP connection
        await self.mcp_client.initialize()
        
        # Load available tools
        await self.tool_manager.load_tools(self.mcp_client)
        
        # Initialize cache
        self.cache_manager.initialize()
        
        logger.info("AudioFab Agent System initialized successfully")
    
    async def process_request(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user request with intelligent optimization.
        
        Args:
            user_input: Natural language request
            context: Optional context information
            
        Returns:
            Comprehensive response with results and metadata
        """
        start_time = datetime.now()
        self.performance_metrics["total_requests"] += 1
        
        try:
            # Generate request cache key
            cache_key = self._generate_cache_key(user_input, context)
            
            # Check cache first
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                self.performance_metrics["cache_hits"] += 1
                logger.info(f"Cache hit for request: {user_input[:50]}...")
                return {**cached_result, "cached": True}
            
            # Analyze request and select tools
            analysis = await self._analyze_request(user_input, context)
            selected_tools = analysis["selected_tools"]
            
            # Optimize execution plan
            execution_plan = await self._create_execution_plan(selected_tools, analysis)
            
            # Execute with monitoring
            results = await self._execute_plan(execution_plan)
            
            # Format response
            response = await self._format_response(results, analysis)
            
            # Cache successful results
            if results.get("success", False):
                await self.cache_manager.set(cache_key, response, ttl_hours=self.cache_ttl_hours)
            
            # Update performance metrics
            self._update_performance_metrics(start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return await self._handle_error(user_input, e)
    
    async def _analyze_request(self, user_input: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user request and select appropriate tools."""
        
        # Create analysis prompt
        tools_summary = self.tool_manager.get_tools_summary()
        
        analysis_prompt = f"""
        Analyze the following audio processing request and select the most appropriate tools:
        
        Request: {user_input}
        Context: {json.dumps(context or {}, indent=2)}
        
        Available tools:
        {json.dumps(tools_summary, indent=2)}
        
        Provide your analysis in the following format:
        {{
            "intent": "main user intent",
            "primary_task": "primary audio processing task",
            "selected_tools": ["tool1", "tool2"],
            "tool_parameters": {{"tool1": {{"param": "value"}}}},
            "execution_order": ["tool1", "tool2"],
            "estimated_complexity": "low|medium|high",
            "suggested_optimizations": ["optimization1", "optimization2"]
        }}
        """
        
        # Use LLM for intelligent analysis
        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": analysis_prompt}],
            model="gpt-4-turbo-preview"
        )
        
        try:
            analysis = json.loads(response.choices[0].message.content)
            return analysis
        except Exception as e:
            logger.warning(f"Failed to parse analysis, using fallback: {e}")
            return await self._fallback_analysis(user_input)
    
    async def _fallback_analysis(self, user_input: str) -> Dict[str, Any]:
        """Fallback analysis when LLM parsing fails."""
        # Simple keyword-based analysis
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ["generate", "create", "make", "produce"]):
            if "music" in user_input_lower:
                return {
                    "intent": "music generation",
                    "primary_task": "generate_music",
                    "selected_tools": ["MusicGenTool"],
                    "execution_order": ["MusicGenTool"],
                    "estimated_complexity": "medium"
                }
            elif "speech" in user_input_lower or "voice" in user_input_lower:
                return {
                    "intent": "speech synthesis",
                    "primary_task": "text_to_speech",
                    "selected_tools": ["CosyVoice2Tool"],
                    "execution_order": ["CosyVoice2Tool"],
                    "estimated_complexity": "low"
                }
        
        return {
            "intent": "general_audio_processing",
            "primary_task": "analyze_and_process",
            "selected_tools": ["FunASRTool"],
            "execution_order": ["FunASRTool"],
            "estimated_complexity": "medium"
        }
    
    async def _create_execution_plan(self, selected_tools: List[str], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimized execution plan."""
        
        plan = {
            "tools": selected_tools,
            "execution_order": analysis.get("execution_order", selected_tools),
            "parallel_groups": self._identify_parallel_groups(selected_tools),
            "retry_config": {
                "max_attempts": self.retry_attempts,
                "delay": self.retry_delay
            },
            "timeout_per_tool": 300,  # 5 minutes
            "estimated_total_time": 0
        }
        
        # Calculate estimated time
        for tool in selected_tools:
            complexity = analysis.get("estimated_complexity", "medium")
            estimated_time = {"low": 30, "medium": 120, "high": 300}.get(complexity, 120)
            plan["estimated_total_time"] += estimated_time
        
        return plan
    
    def _identify_parallel_groups(self, tools: List[str]) -> List[List[str]]:
        """Identify tools that can run in parallel."""
        # Simple heuristic: tools that don't depend on each other's outputs
        independent_tools = ["FunASRTool", "ClearVoiceTool", "MusicGenTool"]
        
        parallel_groups = []
        current_group = []
        
        for tool in tools:
            if tool in independent_tools:
                current_group.append(tool)
            else:
                if current_group:
                    parallel_groups.append(current_group)
                    current_group = []
                parallel_groups.append([tool])
        
        if current_group:
            parallel_groups.append(current_group)
        
        return parallel_groups
    
    async def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the optimized plan."""
        
        results = {
            "success": True,
            "tool_results": {},
            "execution_metadata": {
                "start_time": datetime.now().isoformat(),
                "parallel_groups": plan["parallel_groups"],
                "total_tools": len(plan["tools"])
            }
        }
        
        try:
            # Execute tools in optimized order
            for tool_group in plan["parallel_groups"]:
                if len(tool_group) == 1:
                    # Sequential execution
                    tool = tool_group[0]
                    result = await self._execute_single_tool(tool, plan)
                    results["tool_results"][tool] = result
                    
                    if not result.get("success", False):
                        results["success"] = False
                        break
                else:
                    # Parallel execution
                    parallel_results = await self._execute_parallel_tools(tool_group, plan)
                    results["tool_results"].update(parallel_results)
                    
                    if not all(r.get("success", False) for r in parallel_results.values()):
                        results["success"] = False
                        break
            
            results["execution_metadata"]["end_time"] = datetime.now().isoformat()
            results["execution_metadata"]["duration_seconds"] = (
                datetime.now() - datetime.fromisoformat(results["execution_metadata"]["start_time"])
            ).total_seconds()
            
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            logger.error(f"Execution failed: {e}")
        
        return results
    
    async def _execute_single_tool(self, tool_name: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool with retry logic."""
        
        max_attempts = plan["retry_config"]["max_attempts"]
        delay = plan["retry_config"]["delay"]
        
        for attempt in range(max_attempts):
            try:
                result = await self.execution_engine.execute_tool(tool_name, {})
                if result.get("success", False):
                    return result
                else:
                    logger.warning(f"Tool {tool_name} failed attempt {attempt + 1}")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay * (attempt + 1))
            except Exception as e:
                logger.error(f"Tool {tool_name} error on attempt {attempt + 1}: {e}")
                if attempt == max_attempts - 1:
                    return {"success": False, "error": str(e)}
                await asyncio.sleep(delay * (attempt + 1))
        
        return {"success": False, "error": "Max retry attempts exceeded"}
    
    async def _execute_parallel_tools(self, tools: List[str], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiple tools in parallel."""
        
        async def execute_tool_async(tool_name: str):
            return await self._execute_single_tool(tool_name, plan)
        
        tasks = [execute_tool_async(tool) for tool in tools]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        tool_results = {}
        for tool, result in zip(tools, results):
            if isinstance(result, Exception):
                tool_results[tool] = {"success": False, "error": str(result)}
            else:
                tool_results[tool] = result
        
        return tool_results
    
    async def _format_response(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final response for the user."""
        
        if results["success"]:
            # Create user-friendly summary
            summary_parts = []
            for tool, result in results["tool_results"].items():
                if result.get("success", False):
                    summary_parts.append(f"✅ {tool}: Completed successfully")
                else:
                    summary_parts.append(f"❌ {tool}: {result.get('error', 'Failed')}")
            
            summary = "\n".join(summary_parts)
            
            return {
                "success": True,
                "summary": summary,
                "detailed_results": results["tool_results"],
                "execution_metadata": results["execution_metadata"],
                "analysis": analysis,
                "performance_metrics": self.get_performance_metrics()
            }
        else:
            return {
                "success": False,
                "error": results.get("error", "Execution failed"),
                "detailed_results": results.get("tool_results", {}),
                "execution_metadata": results.get("execution_metadata", {}),
                "analysis": analysis
            }
    
    async def _handle_error(self, user_input: str, error: Exception) -> Dict[str, Any]:
        """Handle errors gracefully with user-friendly messages."""
        
        error_response = {
            "success": False,
            "error": str(error),
            "user_message": "I'm sorry, I encountered an issue processing your request.",
            "suggested_actions": [
                "Check your input format",
                "Verify file paths exist",
                "Try a simpler request",
                "Check system resources"
            ]
        }
        
        # Log error for debugging
        logger.error(f"Request error for '{user_input}': {error}")
        
        return error_response
    
    def _generate_cache_key(self, user_input: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for request caching."""
        cache_data = {
            "input": user_input,
            "context": context or {},
            "tools_version": "1.0.0"
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _update_performance_metrics(self, start_time: datetime):
        """Update performance tracking metrics."""
        duration = (datetime.now() - start_time).total_seconds()
        
        # Update average response time
        total_requests = self.performance_metrics["total_requests"]
        current_avg = self.performance_metrics["average_response_time"]
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + duration) / total_requests
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self.performance_metrics,
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"] / 
                max(self.performance_metrics["total_requests"], 1)
            ),
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check MCP client
        try:
            health_status["components"]["mcp_client"] = {
                "status": "connected" if self.mcp_client.session else "disconnected"
            }
        except Exception as e:
            health_status["components"]["mcp_client"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check tool manager
        try:
            available_tools = len(self.tool_manager.get_available_tools())
            health_status["components"]["tool_manager"] = {
                "status": "ready",
                "available_tools": available_tools
            }
        except Exception as e:
            health_status["components"]["tool_manager"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check cache manager
        try:
            cache_stats = self.cache_manager.get_stats()
            health_status["components"]["cache_manager"] = {
                "status": "ready",
                "stats": cache_stats
            }
        except Exception as e:
            health_status["components"]["cache_manager"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        return health_status