import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Create output directories
OUTPUT_DIR = Path("output")
TENSOR_DIR = OUTPUT_DIR / "tensors"
PLOT_DIR = OUTPUT_DIR / "plots"

for dir_path in [TENSOR_DIR, PLOT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 创建 Tensor 处理 MCP 服务器
mcp = FastMCP("Tensor Processing Tool")

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

@mcp.tool()
def get_gpu_info() -> Dict[str, Any]:
    """获取 GPU 信息。

    Returns:
        包含 GPU 信息的字典
    """
    try:
        if not torch.cuda.is_available():
            return {
                "success": True,
                "available": False,
                "message": "CUDA is not available on this system"
            }
        
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        # 收集所有 GPU 的信息
        devices_info = []
        for i in range(gpu_count):
            device_props = torch.cuda.get_device_properties(i)
            total_memory = device_props.total_memory / (1024 ** 3)  # 转换为 GB
            reserved_memory = torch.cuda.memory_reserved(i) / (1024 ** 3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024 ** 3)
            free_memory = total_memory - allocated_memory
            
            devices_info.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(total_memory, 2),
                "reserved_memory_gb": round(reserved_memory, 2),
                "allocated_memory_gb": round(allocated_memory, 2),
                "free_memory_gb": round(free_memory, 2),
                "compute_capability": f"{device_props.major}.{device_props.minor}"
            })
        
        return {
            "success": True,
            "available": True,
            "count": gpu_count,
            "current_device": current_device,
            "current_device_name": device_name,
            "devices": devices_info
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def set_gpu_device(device_id: int) -> Dict[str, Any]:
    """设置当前使用的 GPU 设备。

    Args:
        device_id: GPU 设备 ID

    Returns:
        操作结果字典
    """
    try:
        if not torch.cuda.is_available():
            return {
                "success": False,
                "error": "CUDA is not available on this system"
            }
            
        if device_id >= torch.cuda.device_count() or device_id < 0:
            return {
                "success": False,
                "error": f"Invalid device ID. Valid range: 0-{torch.cuda.device_count()-1}"
            }
            
        torch.cuda.set_device(device_id)
        device_name = torch.cuda.get_device_name(device_id)
        
        return {
            "success": True,
            "device_id": device_id,
            "device_name": device_name,
            "message": f"Successfully set device to GPU {device_id} ({device_name})"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def load_numpy_file(file_path: str) -> Dict[str, Any]:
    """加载 .npy 格式的 NumPy 数组文件。

    Args:
        file_path: NumPy 文件路径

    Returns:
        包含加载的数组信息的字典
    """
    try:
        # 加载 NumPy 数组
        array = np.load(file_path)
        
        return {
            "success": True,
            "shape": array.shape,
            "dtype": str(array.dtype),
            "size": array.size,
            "min": float(np.min(array)) if array.size > 0 else None,
            "max": float(np.max(array)) if array.size > 0 else None,
            "mean": float(np.mean(array)) if array.size > 0 else None,
            "content_sample": str(array.flatten()[:10]) if array.size > 0 else "[]"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def load_torch_file(file_path: str, to_device: Optional[str] = None) -> Dict[str, Any]:
    """加载 .pth 格式的 PyTorch 张量文件。

    Args:
        file_path: PyTorch 文件路径
        to_device: 加载到的设备 ('cpu' 或 'cuda')

    Returns:
        包含加载的张量信息的字典
    """
    try:
        # 确定设备
        if to_device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(to_device)
        
        # 加载 PyTorch 张量
        tensor = torch.load(file_path, map_location=device)
        
        # 检查加载的对象类型
        if isinstance(tensor, torch.Tensor):
            info = {
                "type": "tensor",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "numel": tensor.numel(),
                "content_sample": str(tensor.flatten()[:10].tolist()) if tensor.numel() > 0 else "[]"
            }
            if tensor.numel() > 0:
                info.update({
                    "min": float(tensor.min()),
                    "max": float(tensor.max()),
                    "mean": float(tensor.mean())
                })
        elif isinstance(tensor, dict):
            # 如果是字典，返回键和每个张量的基本信息
            info = {
                "type": "dict",
                "keys": list(tensor.keys()),
                "contents": {}
            }
            for key, val in tensor.items():
                if isinstance(val, torch.Tensor):
                    info["contents"][str(key)] = {
                        "shape": list(val.shape),
                        "dtype": str(val.dtype),
                        "device": str(val.device)
                    }
                else:
                    info["contents"][str(key)] = str(type(val))
        else:
            info = {
                "type": str(type(tensor))
            }
        
        return {
            "success": True,
            "info": info
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def convert_numpy_to_tensor(numpy_path: str, tensor_path: Optional[str] = None, to_device: Optional[str] = None) -> Dict[str, Any]:
    """将 NumPy 数组转换为 PyTorch 张量并保存。

    Args:
        numpy_path: NumPy 文件路径
        tensor_path: 输出的张量文件路径（可选，默认在相同位置生成）
        to_device: 转换后的设备（'cpu' 或 'cuda'）

    Returns:
        包含转换结果的字典
    """
    try:
        # 加载 NumPy 数组
        array = np.load(numpy_path)
        
        # 确定设备
        if to_device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(to_device)
        
        # 转换为张量
        tensor = torch.from_numpy(array).to(device)
        
        # 确定保存路径
        if tensor_path is None:
            tensor_path = TENSOR_DIR / f"from_numpy_{get_timestamp()}.pth"
        
        # 保存张量
        torch.save(tensor, tensor_path)
        
        return {
            "success": True,
            "numpy_path": numpy_path,
            "tensor_path": str(tensor_path),
            "tensor_shape": list(tensor.shape),
            "tensor_dtype": str(tensor.dtype),
            "tensor_device": str(tensor.device)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def convert_tensor_to_numpy(tensor_path: str, numpy_path: Optional[str] = None) -> Dict[str, Any]:
    """将 PyTorch 张量转换为 NumPy 数组并保存。

    Args:
        tensor_path: PyTorch 张量文件路径
        numpy_path: 输出的 NumPy 文件路径（可选，默认在相同位置生成）

    Returns:
        包含转换结果的字典
    """
    try:
        # 加载张量（始终加载到 CPU，因为 NumPy 只支持 CPU）
        tensor = torch.load(tensor_path, map_location="cpu")
        
        # 检查加载的对象类型
        if not isinstance(tensor, torch.Tensor):
            return {
                "success": False,
                "error": f"Loaded object is not a tensor, but {type(tensor)}"
            }
        
        # 转换为 NumPy
        array = tensor.detach().numpy()
        
        # 确定保存路径
        if numpy_path is None:
            numpy_path = TENSOR_DIR / f"from_tensor_{get_timestamp()}.npy"
        
        # 保存 NumPy 数组
        np.save(numpy_path, array)
        
        return {
            "success": True,
            "tensor_path": tensor_path,
            "numpy_path": str(numpy_path),
            "array_shape": array.shape,
            "array_dtype": str(array.dtype)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def move_tensor_to_device(tensor_path: str, device: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """将张量移动到指定设备（CPU 或 CUDA）。

    Args:
        tensor_path: 张量文件路径
        device: 目标设备 ('cpu' 或 'cuda[:id]')
        output_path: 输出文件路径（可选）

    Returns:
        包含操作结果的字典
    """
    try:
        # 加载张量
        tensor = torch.load(tensor_path, map_location="cpu")
        
        # 检查加载的对象类型
        if not isinstance(tensor, torch.Tensor):
            return {
                "success": False,
                "error": f"Loaded object is not a tensor, but {type(tensor)}"
            }
        
        # 移动到目标设备
        device = torch.device(device)
        tensor = tensor.to(device)
        
        # 确定保存路径
        if output_path is None:
            output_path = TENSOR_DIR / f"moved_to_{device}_{get_timestamp()}.pth"
        
        # 保存张量
        torch.save(tensor, output_path)
        
        return {
            "success": True,
            "original_path": tensor_path,
            "output_path": str(output_path),
            "device": str(tensor.device),
            "shape": list(tensor.shape)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def concatenate_tensors(tensor_paths: List[str], dim: int = 0, output_path: Optional[str] = None) -> Dict[str, Any]:
    """沿指定维度连接多个张量。

    Args:
        tensor_paths: 张量文件路径列表
        dim: 连接的维度
        output_path: 输出文件路径（可选）

    Returns:
        包含操作结果的字典
    """
    try:
        # 加载所有张量
        tensors = []
        for path in tensor_paths:
            tensor = torch.load(path, map_location="cpu")
            # 检查加载的对象类型
            if not isinstance(tensor, torch.Tensor):
                return {
                    "success": False,
                    "error": f"Loaded object from {path} is not a tensor, but {type(tensor)}"
                }
            tensors.append(tensor)
        
        # 连接张量
        result = torch.cat(tensors, dim=dim)
        
        # 确定保存路径
        if output_path is None:
            output_path = TENSOR_DIR / f"concatenated_dim{dim}_{get_timestamp()}.pth"
        
        # 保存结果
        torch.save(result, output_path)
        
        return {
            "success": True,
            "input_paths": tensor_paths,
            "output_path": str(output_path),
            "concatenation_dim": dim,
            "result_shape": list(result.shape),
            "input_shapes": [list(t.shape) for t in tensors]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def split_tensor(tensor_path: str, dim: int = 0, sections: Union[int, List[int]] = 2, 
                output_dir: Optional[str] = None) -> Dict[str, Any]:
    """沿指定维度拆分张量。

    Args:
        tensor_path: 张量文件路径
        dim: 拆分的维度
        sections: 拆分方式
            - 如果是整数：拆分为等大小的块
            - 如果是列表：按照列表中的索引拆分
        output_dir: 输出目录（可选）

    Returns:
        包含操作结果的字典
    """
    try:
        # 加载张量
        tensor = torch.load(tensor_path, map_location="cpu")
        
        # 检查加载的对象类型
        if not isinstance(tensor, torch.Tensor):
            return {
                "success": False,
                "error": f"Loaded object is not a tensor, but {type(tensor)}"
            }
        
        # 确定输出目录
        if output_dir is None:
            timestamp = get_timestamp()
            output_dir = TENSOR_DIR / f"split_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
        
        # 拆分张量
        if isinstance(sections, int):
            # 等大小拆分
            result = torch.chunk(tensor, sections, dim=dim)
        else:
            # 按照索引拆分
            result = torch.split(tensor, sections, dim=dim)
        
        # 保存结果
        output_paths = []
        for i, t in enumerate(result):
            out_path = os.path.join(output_dir, f"part_{i}.pth")
            torch.save(t, out_path)
            output_paths.append(out_path)
        
        return {
            "success": True,
            "input_path": tensor_path,
            "output_dir": str(output_dir),
            "output_paths": output_paths,
            "split_dim": dim,
            "sections": sections,
            "num_parts": len(result),
            "result_shapes": [list(t.shape) for t in result]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def save_tensor(tensor_data: Dict[str, Any], output_path: Optional[str] = None) -> Dict[str, Any]:
    """保存张量数据到 PyTorch .pth 文件。

    Args:
        tensor_data: 张量数据字典，包含以下字段：
            - "shape": 张量形状列表
            - "values": 张量值列表（可选，如果不提供，则生成随机值）
            - "dtype": 张量数据类型（可选，默认 "float32"）
        output_path: 输出文件路径（可选）

    Returns:
        包含操作结果的字典
    """
    try:
        # 解析张量数据
        shape = tensor_data.get("shape")
        if shape is None:
            return {
                "success": False,
                "error": "Missing required 'shape' in tensor_data"
            }
        
        # 获取数据类型
        dtype_str = tensor_data.get("dtype", "float32")
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "uint8": torch.uint8,
            "bool": torch.bool
        }
        
        if dtype_str not in dtype_map:
            return {
                "success": False,
                "error": f"Unsupported dtype: {dtype_str}. Supported dtypes: {list(dtype_map.keys())}"
            }
        
        dtype = dtype_map[dtype_str]
        
        # 创建张量
        values = tensor_data.get("values")
        if values is not None:
            tensor = torch.tensor(values, dtype=dtype).reshape(shape)
        else:
            # 生成随机值
            tensor = torch.randn(shape, dtype=dtype)
        
        # 确定保存路径
        if output_path is None:
            output_path = TENSOR_DIR / f"tensor_{get_timestamp()}.pth"
        
        # 保存张量
        torch.save(tensor, output_path)
        
        return {
            "success": True,
            "output_path": str(output_path),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "is_random": values is None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def tensor_operations(tensor_path: str, operation: str, params: Optional[Dict[str, Any]] = None, 
                     output_path: Optional[str] = None) -> Dict[str, Any]:
    """对张量执行基本操作。

    Args:
        tensor_path: 张量文件路径
        operation: 操作类型，支持：
            - "reshape": 改变形状
            - "transpose": 转置维度
            - "clone": 克隆张量
            - "add": 加法操作
            - "multiply": 乘法操作
            - "mean": 求均值
            - "sum": 求和
            - "norm": 求范数
        params: 操作参数（可选）
        output_path: 输出文件路径（可选）

    Returns:
        包含操作结果的字典
    """
    try:
        # 加载张量
        tensor = torch.load(tensor_path, map_location="cpu")
        
        # 检查加载的对象类型
        if not isinstance(tensor, torch.Tensor):
            return {
                "success": False,
                "error": f"Loaded object is not a tensor, but {type(tensor)}"
            }
        
        params = params or {}
        
        # 执行操作
        if operation == "reshape":
            shape = params.get("shape")
            if shape is None:
                return {"success": False, "error": "Missing 'shape' parameter for reshape operation"}
            result = tensor.reshape(shape)
            
        elif operation == "transpose":
            dim0 = params.get("dim0", 0)
            dim1 = params.get("dim1", 1)
            result = tensor.transpose(dim0, dim1)
            
        elif operation == "clone":
            result = tensor.clone()
            
        elif operation == "add":
            value = params.get("value", 0)
            result = tensor + value
            
        elif operation == "multiply":
            value = params.get("value", 1)
            result = tensor * value
            
        elif operation == "mean":
            dim = params.get("dim")
            if dim is not None:
                result = tensor.mean(dim=dim)
            else:
                result = tensor.mean()
                
        elif operation == "sum":
            dim = params.get("dim")
            if dim is not None:
                result = tensor.sum(dim=dim)
            else:
                result = tensor.sum()
                
        elif operation == "norm":
            p = params.get("p", 2)
            dim = params.get("dim")
            if dim is not None:
                result = tensor.norm(p=p, dim=dim)
            else:
                result = tensor.norm(p=p)
                
        else:
            return {
                "success": False,
                "error": f"Unsupported operation: {operation}"
            }
        
        # 如果结果是标量值，则直接返回
        if isinstance(result, torch.Tensor) and result.numel() == 1:
            return {
                "success": True,
                "operation": operation,
                "input_path": tensor_path,
                "scalar_result": float(result.item())
            }
        
        # 确定保存路径
        if output_path is None:
            output_path = TENSOR_DIR / f"{operation}_{get_timestamp()}.pth"
        
        # 保存结果
        torch.save(result, output_path)
        
        return {
            "success": True,
            "operation": operation,
            "input_path": tensor_path,
            "output_path": str(output_path),
            "params": params,
            "result_shape": list(result.shape) if isinstance(result, torch.Tensor) else None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # 初始化并运行服务器
    mcp.run()
