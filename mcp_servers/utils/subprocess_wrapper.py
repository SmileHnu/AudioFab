"""
Standardized subprocess execution wrapper for MCP tools.

This module provides a consistent way to handle subprocess execution with
proper success/failure detection and standardized return value formats.
"""

import subprocess
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class SubprocessResult:
    """Standardized result container for subprocess execution."""
    
    def __init__(self, success: bool, return_code: int, stdout: str, stderr: str,
                 output_files: Optional[List[str]] = None, error_message: Optional[str] = None):
        self.success = success
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.output_files = output_files or []
        self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to standardized dictionary format."""
        result = {
            "success": self.success,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "output_files": self.output_files
        }
        
        if self.error_message:
            result["error"] = self.error_message
            
        return result


class SubprocessWrapper:
    """Wrapper for subprocess execution with standardized error handling."""
    
    @staticmethod
    def run_command(
        cmd: List[str],
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
        check_output_files: bool = True,
        expected_files: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ) -> SubprocessResult:
        """
        Execute a subprocess command with standardized error handling.
        
        Args:
            cmd: Command to execute as list of strings
            cwd: Working directory for the command
            timeout: Timeout in seconds
            check_output_files: Whether to check for expected output files
            expected_files: List of expected output file patterns/paths
            output_dir: Directory to check for output files
            env: Environment variables to set
            
        Returns:
            SubprocessResult with standardized success/failure detection
        """
        try:
            # Execute the command
            process = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            
            # Determine success criteria
            success = process.returncode == 0
            
            # Check for actual errors vs informational stderr
            actual_error = None
            if process.returncode != 0:
                actual_error = f"Command failed with exit code {process.returncode}: {process.stderr}"
            elif process.stderr.strip() and not process.stdout.strip():
                # Only stderr with no stdout might indicate an issue
                actual_error = f"Command produced only stderr output: {process.stderr}"
            
            # Find output files
            output_files = []
            if check_output_files and output_dir and os.path.exists(output_dir):
                output_path = Path(output_dir)
                
                if expected_files:
                    # Look for specific expected files
                    for pattern in expected_files:
                        matches = list(output_path.glob(pattern))
                        output_files.extend(str(f) for f in matches)
                else:
                    # Default: look for common output file types
                    common_extensions = ['*.mp4', '*.wav', '*.mp3', '*.png', '*.jpg', '*.txt', '*.json']
                    for ext in common_extensions:
                        matches = list(output_path.glob(ext))
                        output_files.extend(str(f) for f in matches)
            
            # Override success if we have output files but returncode != 0
            # This handles cases where tools exit with non-zero codes but still produce valid output
            if not success and output_files:
                # Check if stderr contains warnings rather than errors
                stderr_lower = process.stderr.lower()
                error_keywords = ['error', 'fatal', 'exception', 'failed', 'abort', 'crash', 'abort']
                has_real_error = any(keyword in stderr_lower for keyword in error_keywords)
                
                # Also check stdout for error indicators
                stdout_lower = process.stdout.lower()
                has_stdout_error = any(keyword in stdout_lower for keyword in error_keywords)
                
                if not has_real_error and not has_stdout_error:
                    success = True
                    actual_error = None
            
            return SubprocessResult(
                success=success,
                return_code=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                output_files=output_files,
                error_message=actual_error
            )
            
        except subprocess.TimeoutExpired as e:
            return SubprocessResult(
                success=False,
                return_code=-1,
                stdout=e.stdout.decode() if e.stdout else "",
                stderr=e.stderr.decode() if e.stderr else "",
                error_message=f"Command timed out after {timeout} seconds"
            )
        except FileNotFoundError as e:
            return SubprocessResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=str(e),
                error_message=f"Command not found: {e}"
            )
        except Exception as e:
            return SubprocessResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=str(e),
                error_message=f"Unexpected error: {str(e)}"
            )


class MCPSubprocessTool:
    """Base class for MCP tools that use subprocess execution."""
    
    @staticmethod
    def format_success_result(
        result: SubprocessResult,
        tool_name: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format a successful subprocess result into standardized MCP format.
        
        Args:
            result: SubprocessResult from execution
            tool_name: Name of the tool for identification
            additional_data: Additional data to include in response
            
        Returns:
            Standardized MCP response dictionary
        """
        response = {
            "success": True,
            "tool_name": tool_name,
            "processing_info": {
                "return_code": result.return_code,
                "stdout_preview": result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout,
                "stderr_preview": result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr,
                "output_files": result.output_files
            }
        }
        
        if additional_data:
            response.update(additional_data)
            
        # Add primary output file if available
        if result.output_files:
            response["output_path"] = result.output_files[0]
            
        return response
    
    @staticmethod
    def format_error_result(
        result: SubprocessResult,
        tool_name: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format a failed subprocess result into standardized MCP format.
        
        Args:
            result: SubprocessResult from execution
            tool_name: Name of the tool for identification
            context: Additional context about the failure
            
        Returns:
            Standardized MCP error response dictionary
        """
        error_msg = result.error_message or "Command execution failed"
        if context:
            error_msg = f"{context}: {error_msg}"
            
        return {
            "success": False,
            "tool_name": tool_name,
            "error": error_msg,
            "details": {
                "return_code": result.return_code,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        }


def validate_tool_inputs(inputs: Dict[str, Any], required: List[str]) -> Optional[str]:
    """
    Validate that required inputs are provided.
    
    Args:
        inputs: Dictionary of input parameters
        required: List of required parameter names
        
    Returns:
        Error message if validation fails, None if valid
    """
    missing = [param for param in required if not inputs.get(param)]
    if missing:
        return f"Missing required parameters: {', '.join(missing)}"
    
    # Check file existence
    file_params = [k for k, v in inputs.items() if isinstance(v, str) and 
                   (v.endswith(('.mp4', '.wav', '.mp3', '.jpg', '.png', '.txt')) or 
                    os.path.sep in v)]
    
    for param in file_params:
        file_path = inputs[param]
        if file_path and not os.path.exists(file_path):
            return f"File not found: {file_path} (parameter: {param})"
    
    return None