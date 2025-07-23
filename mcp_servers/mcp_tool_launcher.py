#!/usr/bin/env python
import sys
import json
import traceback
import argparse
import importlib.util
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_file_path(file_path: str, file_type: str) -> bool:
    """验证文件路径是否存在且可访问"""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"{file_type} file not found: {file_path}")
        return False
    if not path.is_file():
        logger.error(f"{file_type} path is not a file: {file_path}")
        return False
    return True

def load_params(params_file: str) -> Optional[Dict[str, Any]]:
    """加载并验证参数文件"""
    try:
        if not validate_file_path(params_file, "Parameters"):
            return None
        
        with open(params_file, 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        if not isinstance(params, dict):
            logger.error("Parameters file must contain a JSON object")
            return None
            
        return params
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in parameters file: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading parameters file: {e}")
        return None

def load_tool_module(module_path: str) -> Optional[Any]:
    """加载工具模块"""
    try:
        if not validate_file_path(module_path, "Module"):
            return None
            
        module_dir = str(Path(module_path).parent)
        module_name = Path(module_path).stem
        
        # 添加模块目录到系统路径
        if module_dir not in sys.path:
            sys.path.append(module_dir)
        
        # 尝试导入模块
        try:
            # 先尝试直接导入
            tool_module = __import__(module_name)
            logger.info(f"Successfully imported module {module_name}")
            return tool_module
        except ImportError:
            # 如果直接导入失败，尝试使用importlib加载
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None:
                logger.error(f"Could not find module spec for {module_name}")
                return None
                
            tool_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tool_module)
            logger.info(f"Successfully loaded module {module_name} using importlib")
            return tool_module
            
    except Exception as e:
        logger.error(f"Error loading module: {e}")
        return None

def get_tool_function(module: Any, tool_name: str) -> Optional[callable]:
    """获取工具函数"""
    try:
        if not hasattr(module, tool_name):
            logger.error(f"Tool function '{tool_name}' not found in module")
            return None
            
        tool_func = getattr(module, tool_name)
        if not callable(tool_func):
            logger.error(f"'{tool_name}' is not a callable function")
            return None
            
        return tool_func
    except Exception as e:
        logger.error(f"Error getting tool function: {e}")
        return None

def execute_tool(tool_func: callable, params: Dict[str, Any]) -> Dict[str, Any]:
    """执行工具函数"""
    try:
        result = tool_func(**params)
        
        # 如果结果不是字典，转换为字符串
        if not isinstance(result, dict):
            logger.error("Tool function must return a dictionary")
            return {
                "success": False,
                "error": "Tool function returned invalid result type"
            }
            
        # 确保结果包含 success 字段
        if "success" not in result:
            result["success"] = True
            
        return result
    except TypeError as e:
        logger.error(f"Invalid parameters for tool function: {e}")
        return {
            "success": False,
            "error": f"Invalid parameters: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Error executing tool: {e}")
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def main():
    """
    通用工具启动器 - 加载并执行指定的工具
    
    参数:
      --tool_name: 要执行的工具名称
      --module_path: 工具模块路径
      --params_file: 包含参数的JSON文件路径
    """
    parser = argparse.ArgumentParser(description="MCP工具启动器")
    parser.add_argument("--tool_name", type=str, required=True, help="要执行的工具名称")
    parser.add_argument("--module_path", type=str, required=True, help="工具模块路径")
    parser.add_argument("--params_file", type=str, required=True, help="参数文件路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    try:
        # 加载参数
        params = load_params(args.params_file)
        if params is None:
            sys.exit(1)
            
        # 加载模块
        tool_module = load_tool_module(args.module_path)
        if tool_module is None:
            sys.exit(1)
            
        # 获取工具函数
        tool_func = get_tool_function(tool_module, args.tool_name)
        if tool_func is None:
            sys.exit(1)
            
        # 执行工具
        result = execute_tool(tool_func, params)
        
        # 输出结果
        print(json.dumps(result, ensure_ascii=False))
        
            
    except KeyboardInterrupt:
        logger.info("Tool execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        error_result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()
