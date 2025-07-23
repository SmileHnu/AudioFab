"""
Standardized return value format utilities for MCP tools.

This module provides decorators and utility functions to ensure consistent
return value formats across all MCP tools in the AudioFab system.
"""

import functools
import traceback
from typing import Dict, Any, Optional, Callable
from pathlib import Path


def standardized_return(func: Callable) -> Callable:
    """
    Decorator to ensure consistent return value format for MCP tools.
    
    This decorator wraps tool functions to ensure they always return
    a standardized dictionary format with proper success/failure indicators.
    
    Expected format:
    {
        "success": bool,
        "error": str (optional, only if success=False),
        "traceback": str (optional, for debugging),
        ... (additional tool-specific data)
    }
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Dict[str, Any]:
        try:
            result = func(*args, **kwargs)
            
            # If result is already a dict, ensure it has success key
            if isinstance(result, dict):
                if "success" not in result:
                    # Assume success if no explicit failure
                    result["success"] = "error" not in result
                return result
            
            # If result is a string/path, wrap it in standard format
            elif isinstance(result, (str, Path)):
                return {
                    "success": True,
                    "output_path": str(result)
                }
            
            # For other types, create a basic success response
            else:
                return {
                    "success": True,
                    "result": result
                }
                
        except FileNotFoundError as e:
            return {
                "success": False,
                "error": f"Required file not found: {str(e)}",
                "traceback": traceback.format_exc()
            }
        except ValueError as e:
            return {
                "success": False,
                "error": f"Invalid parameter: {str(e)}",
                "traceback": traceback.format_exc()
            }
        except PermissionError as e:
            return {
                "success": False,
                "error": f"Permission denied: {str(e)}",
                "traceback": traceback.format_exc()
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    return wrapper


def format_success_response(
    tool_name: str,
    output_path: Optional[str] = None,
    additional_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized success response.
    
    Args:
        tool_name: Name of the tool for identification
        output_path: Primary output file path (optional)
        additional_data: Additional tool-specific data (optional)
    
    Returns:
        Standardized success response dictionary
    """
    response = {
        "success": True,
        "tool_name": tool_name
    }
    
    if output_path:
        response["output_path"] = str(output_path)
    
    if additional_data:
        response.update(additional_data)
    
    return response


def format_error_response(
    tool_name: str,
    error_message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        tool_name: Name of the tool for identification
        error_message: Human-readable error message
        details: Additional error details (optional)
    
    Returns:
        Standardized error response dictionary
    """
    response = {
        "success": False,
        "tool_name": tool_name,
        "error": error_message
    }
    
    if details:
        response["details"] = details
    
    return response


def validate_file_path(file_path: str, param_name: str) -> Optional[str]:
    """
    Validate that a file exists and is accessible.
    
    Args:
        file_path: Path to validate
        param_name: Parameter name for error messages
    
    Returns:
        Error message if validation fails, None if valid
    """
    from pathlib import Path
    
    if not file_path:
        return f"Parameter '{param_name}' is required"
    
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path} (parameter: {param_name})"
    
    if not path.is_file():
        return f"Path is not a file: {file_path} (parameter: {param_name})"
    
    return None


def validate_directory_path(dir_path: str, param_name: str) -> Optional[str]:
    """
    Validate that a directory exists and is accessible.
    
    Args:
        dir_path: Directory path to validate
        param_name: Parameter name for error messages
    
    Returns:
        Error message if validation fails, None if valid
    """
    from pathlib import Path
    
    if not dir_path:
        return f"Parameter '{param_name}' is required"
    
    path = Path(dir_path)
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return f"Cannot create directory {dir_path}: {str(e)}"
    
    if not path.is_dir():
        return f"Path is not a directory: {dir_path} (parameter: {param_name})"
    
    return None