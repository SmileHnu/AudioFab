# MCP Tool Return Value Standardization Guide

## Overview

This document describes the standardized return value system for MCP (Model Context Protocol) tools in the AudioFab system. The system addresses common issues with false positive failures and inconsistent return value formats across different tool types.

## Problems Solved

### 1. False Positive Failures
**Issue**: Subprocess-based tools would report failure when stderr contains content, even if:
- Exit code is 0 (successful execution)
- Valid output files are produced
- Stderr contains only warnings or informational messages

**Solution**: Intelligent error detection that considers:
- Exit code (primary indicator)
- Presence of output files
- Content analysis of stderr (error keywords vs warnings)

### 2. Inconsistent Return Formats
**Issue**: Different tools return different formats:
- API tools: `{"success": True, "result": ...}`
- Subprocess tools: Raise exceptions on failure
- Direct execution: Return strings or paths directly

**Solution**: Standardized dictionary format with consistent keys:
```python
{
    "success": bool,
    "error": str (optional, only if success=False),
    "traceback": str (optional, for debugging),
    "output_path": str (optional, primary output file),
    "output_files": list (optional, all output files),
    "processing_info": dict (optional, execution details),
    ... (tool-specific data)
}
```

## Standardized Components

### 1. SubprocessWrapper (`utils/subprocess_wrapper.py`)

Provides intelligent subprocess execution with proper success/failure detection.

**Usage Example:**
```python
from utils.subprocess_wrapper import SubprocessWrapper, MCPSubprocessTool

# Execute command with standardized handling
result = SubprocessWrapper.run_command(
    cmd=["python", "script.py", "-i", "input.mp4"],
    output_dir="/path/to/output",
    expected_files=["*.mp4"],
    check_output_files=True
)

if result.success:
    # Format successful result
    return MCPSubprocessTool.format_success_result(
        result, 
        "MyTool",
        {"processing_time": "5.2 seconds"}
    )
else:
    # Format error result
    return MCPSubprocessTool.format_error_result(
        result,
        "MyTool",
        "Processing failed"
    )
```

### 2. Standard Format Utilities (`utils/standard_format.py`)

Provides decorators and utility functions for consistent formatting.

**Decorator Usage:**
```python
from utils.standard_format import standardized_return, format_success_response

@standardized_return
def my_tool_function(input_file: str) -> dict:
    # Tool implementation
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    return format_success_response(
        "MyTool",
        output_path="result.mp4",
        {"additional_info": "processing complete"}
    )
```

### 3. Input Validation

```python
from utils.standard_format import validate_file_path, validate_directory_path

# Validate required inputs
error = validate_file_path(input_file, "input_file")
if error:
    return format_error_response("MyTool", error)
```

## Migration Guide

### For Existing Subprocess Tools

**Before:**
```python
import subprocess

process = subprocess.run(cmd, capture_output=True, text=True)
if process.returncode != 0:
    raise RuntimeError(f"Failed: {process.stderr}")

# Manual file checking
files = list(Path(output_dir).glob("*.mp4"))
if not files:
    raise RuntimeError("No output files")
```

**After:**
```python
from utils.subprocess_wrapper import SubprocessWrapper, MCPSubprocessTool

result = SubprocessWrapper.run_command(
    cmd=cmd,
    output_dir=str(output_dir),
    expected_files=["*.mp4"]
)

if result.success:
    return MCPSubprocessTool.format_success_result(
        result, "ToolName", {"custom_data": "value"}
    )
else:
    return MCPSubprocessTool.format_error_result(result, "ToolName")
```

### For API Tools

**Before:**
```python
# Mixed return patterns
return {"result": api_response}  # or
return api_response  # or
raise Exception("Error")
```

**After:**
```python
from utils.standard_format import standardized_return, format_success_response

@standardized_return
def api_tool():
    try:
        result = api_client.call_method()
        return format_success_response("APITool", result["output_path"])
    except Exception as e:
        # Decorator handles exception formatting
        raise e
```

## Success/Failure Detection Logic

### Primary Criteria (in order)
1. **Exit code**: 0 = success, non-zero = potential failure
2. **Output files**: Presence of expected output files indicates success
3. **Stderr content analysis**: Distinguish between warnings and actual errors

### Error Classification
- **Real errors**: Exit code != 0 AND stderr contains error keywords
- **Warnings**: Exit code = 0, stderr contains only warnings, valid output exists
- **False positives**: Exit code != 0 but valid output files exist and stderr contains only warnings

### Error Keywords
The system checks stderr for these keywords to distinguish real errors:
- "error", "fatal", "exception", "failed", "abort", "crash"

## Standard Response Formats

### Success Response
```json
{
    "success": true,
    "tool_name": "ToolName",
    "output_path": "/path/to/output.mp4",
    "output_files": ["/path/to/file1.mp4", "/path/to/file2.txt"],
    "processing_info": {
        "return_code": 0,
        "stdout_preview": "Command output...",
        "stderr_preview": "Warnings if any..."
    },
    "tool_specific_data": {...}
}
```

### Error Response
```json
{
    "success": false,
    "tool_name": "ToolName",
    "error": "Human-readable error message",
    "traceback": "Full traceback for debugging",
    "details": {
        "return_code": 1,
        "stdout": "Command stdout...",
        "stderr": "Error details..."
    }
}
```

## Testing

### Test Cases for Subprocess Tools
1. **Successful execution**: Exit code 0, valid output files
2. **Warning case**: Exit code 0, stderr has warnings, valid output files
3. **False positive**: Exit code 1, stderr has warnings only, valid output files
4. **Real failure**: Exit code 1, stderr has error keywords, no output files
5. **Timeout**: Command exceeds timeout limit
6. **Missing dependencies**: Command not found or file not found

### Test Cases for API Tools
1. **Successful API call**: Valid response from API
2. **API error**: HTTP errors, authentication failures
3. **Invalid parameters**: Parameter validation failures
4. **Network issues**: Connection timeouts, service unavailable

## Implementation Status

- ✅ SubprocessWrapper created (`utils/subprocess_wrapper.py`)
- ✅ Standard format utilities created (`utils/standard_format.py`)
- ✅ hello2_processor.py updated to use new system
- 📋 Other processors to be updated as needed
- 📋 Documentation provided (this file)

## Future Enhancements

1. **Logging integration**: Add structured logging for all tool executions
2. **Progress reporting**: Real-time progress updates for long-running tools
3. **Retry mechanisms**: Automatic retry for transient failures
4. **Health checks**: Tool availability and dependency verification
5. **Metrics collection**: Performance and usage metrics

## Best Practices

1. **Always use decorators**: Apply `@standardized_return` to all tool functions
2. **Validate inputs**: Use validation utilities before processing
3. **Provide meaningful error messages**: Include context and potential solutions
4. **Include processing info**: Add relevant metadata to responses
5. **Handle all exceptions**: Never let unhandled exceptions propagate
6. **Test edge cases**: Test with invalid inputs, missing files, etc.