#!/usr/bin/env python3
"""
Test script for the standardized return value system.

This script tests the new subprocess wrapper and standard format utilities
to ensure they correctly handle various success/failure scenarios.
"""

import os
import tempfile
import subprocess
from pathlib import Path

# Add the mcp_servers directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.subprocess_wrapper import SubprocessWrapper, MCPSubprocessTool
from utils.standard_format import standardized_return, format_success_response, format_error_response, validate_file_path


def test_successful_subprocess():
    """Test successful subprocess execution."""
    print("Test 1: Successful subprocess execution")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple Python script that creates output
        script_content = """
import sys
import os
# Create an output file
with open(os.path.join(sys.argv[2], 'output.txt'), 'w') as f:
    f.write('Hello World')
print('Success message')
"""
        script_path = os.path.join(temp_dir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        cmd = [sys.executable, script_path, 'arg1', temp_dir]
        result = SubprocessWrapper.run_command(
            cmd=cmd,
            output_dir=temp_dir,
            expected_files=["*.txt"]
        )
        
        print(f"  Success: {result.success}")
        print(f"  Return code: {result.return_code}")
        print(f"  Output files: {result.output_files}")
        print(f"  Stdout: {result.stdout.strip()}")
        print(f"  Stderr: {result.stderr.strip()}")
        
        assert result.success, "Should be successful"
        assert len(result.output_files) == 1, "Should have one output file"
        assert "output.txt" in result.output_files[0], "Should create output.txt"
        print("  ✓ PASSED\n")


def test_warning_subprocess():
    """Test subprocess with warnings in stderr but successful execution."""
    print("Test 2: Subprocess with warnings")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a script that produces warnings but succeeds
        script_content = """
import sys
import os
import warnings
warnings.warn('This is just a warning')
with open(os.path.join(sys.argv[2], 'output.txt'), 'w') as f:
    f.write('Hello with warnings')
print('Success with warnings')
"""
        script_path = os.path.join(temp_dir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        cmd = [sys.executable, script_path, 'arg1', temp_dir]
        result = SubprocessWrapper.run_command(
            cmd=cmd,
            output_dir=temp_dir,
            expected_files=["*.txt"]
        )
        
        print(f"  Success: {result.success}")
        print(f"  Return code: {result.return_code}")
        print(f"  Output files: {result.output_files}")
        print(f"  Stderr: {result.stderr.strip()}")
        
        assert result.success, "Should be successful despite warnings"
        assert len(result.output_files) == 1, "Should have one output file"
        print("  ✓ PASSED\n")


def test_real_failure():
    """Test real subprocess failure."""
    print("Test 3: Real subprocess failure")
    
    cmd = [sys.executable, "nonexistent_script.py"]
    result = SubprocessWrapper.run_command(cmd)
    
    print(f"  Success: {result.success}")
    print(f"  Return code: {result.return_code}")
    print(f"  Error message: {result.error_message}")
    print(f"  Output files: {result.output_files}")
    
    assert not result.success, "Should be failed"
    assert result.error_message is not None, "Should have error message"
    print("  ✓ PASSED\n")


def test_false_positive_detection():
    """Test false positive detection (exit code != 0 but valid output)."""
    print("Test 4: False positive detection")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a script that exits with non-zero code but produces output
        script_content = """
import sys
import os
# Create output file
with open(os.path.join(sys.argv[2], 'output.txt'), 'w') as f:
    f.write('Output created despite non-zero exit')
print('Warning: Non-zero exit for testing')
sys.exit(1)  # Non-zero exit code
"""
        script_path = os.path.join(temp_dir, 'test_script.py')
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        cmd = [sys.executable, script_path, 'arg1', temp_dir]
        result = SubprocessWrapper.run_command(
            cmd=cmd,
            output_dir=temp_dir,
            expected_files=["*.txt"]
        )
        
        print(f"  Success: {result.success}")
        print(f"  Return code: {result.return_code}")
        print(f"  Output files: {result.output_files}")
        print(f"  Error message: {result.error_message}")
        
        # This should be treated as success because output files exist and stderr is not an error
        assert result.success, "Should be successful (false positive corrected)"
        assert len(result.output_files) == 1, "Should have one output file"
        print("  ✓ PASSED\n")


def test_format_success_response():
    """Test success response formatting."""
    print("Test 5: Success response formatting")
    
    response = format_success_response(
        "TestTool",
        output_path="/path/to/output.mp4",
        additional_data={"processing_time": "2.5s", "quality": "high"}
    )
    
    print(f"  Response: {response}")
    
    assert response["success"] is True
    assert response["tool_name"] == "TestTool"
    assert response["output_path"] == "/path/to/output.mp4"
    assert response["processing_time"] == "2.5s"
    print("  ✓ PASSED\n")


def test_format_error_response():
    """Test error response formatting."""
    print("Test 6: Error response formatting")
    
    response = format_error_response(
        "TestTool",
        "Test error message",
        {"return_code": 1, "stderr": "error details"}
    )
    
    print(f"  Response: {response}")
    
    assert response["success"] is False
    assert response["tool_name"] == "TestTool"
    assert response["error"] == "Test error message"
    assert response["details"]["return_code"] == 1
    print("  ✓ PASSED\n")


def test_standardized_decorator():
    """Test the standardized_return decorator."""
    print("Test 7: Standardized return decorator")
    
    @standardized_return
    def successful_function():
        return {"output_path": "test.mp4", "metadata": {"duration": 10}}
    
    @standardized_return
    def failing_function():
        raise ValueError("Test error")
    
    @standardized_return
    def string_return_function():
        return "output.mp4"
    
    success_result = successful_function()
    fail_result = failing_function()
    string_result = string_return_function()
    
    print(f"  Success result: {success_result}")
    print(f"  Fail result: {fail_result}")
    print(f"  String result: {string_result}")
    
    assert success_result["success"] is True
    assert success_result["output_path"] == "test.mp4"
    assert fail_result["success"] is False
    assert "error" in fail_result
    assert string_result["success"] is True
    assert string_result["output_path"] == "output.mp4"
    print("  ✓ PASSED\n")


def test_validate_file_path():
    """Test file path validation."""
    print("Test 8: File path validation")
    
    # Test with existing file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("test")
        temp_path = temp_file.name
    
    try:
        error = validate_file_path(temp_path, "test_file")
        print(f"  Existing file validation: {error}")
        assert error is None, "Should pass validation for existing file"
        
        # Test with non-existent file
        error = validate_file_path("/nonexistent/file.txt", "test_file")
        print(f"  Non-existent file validation: {error}")
        assert error is not None, "Should fail validation for non-existent file"
        
        # Test with empty path
        error = validate_file_path("", "test_file")
        print(f"  Empty path validation: {error}")
        assert error is not None, "Should fail validation for empty path"
        
    finally:
        os.unlink(temp_path)
    
    print("  ✓ PASSED\n")


def run_all_tests():
    """Run all tests."""
    print("Running Standardized Return Value System Tests\n")
    print("=" * 50)
    
    try:
        test_successful_subprocess()
        test_warning_subprocess()
        test_real_failure()
        test_false_positive_detection()
        test_format_success_response()
        test_format_error_response()
        test_standardized_decorator()
        test_validate_file_path()
        
        print("=" * 50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The standardized return value system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()