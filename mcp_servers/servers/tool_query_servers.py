import json
import os
from pathlib import Path
from typing import Dict, Any
import numpy as np
from functools import lru_cache

from mcp.server.fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("Tool Query Processor")

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
# Tool description file path （all:Local+API）
# TOOL_DESCRIPTIONS_PATH: Path = PROJECT_ROOT  / "tool_descriptions.json"

# Tool description file path （API:ONLY API）
TOOL_DESCRIPTIONS_PATH: Path = PROJECT_ROOT / "tool_decs" / "tool_descriptions_api.json"


# 全局变量，用于存储句向量模型和工具描述的嵌入向量
_model = None
_tool_embeddings = {}

# 本地预训练模型路径
LOCAL_MODEL_PATH = "/home/chengz/LAMs/pre_train_models/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"


def load_tool_descriptions() -> Dict[str, Any]:
    """加载工具描述文件"""
    try:
        if not TOOL_DESCRIPTIONS_PATH.exists():
            return {}
        with open(TOOL_DESCRIPTIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"加载工具描述出错: {e}")
        return {}


@lru_cache(maxsize=1)
def get_embedding_model():
    """加载并返回句子嵌入模型，使用缓存确保只加载一次"""
    try:
        from sentence_transformers import SentenceTransformer
        print("正在加载句向量模型...")
        # 使用本地预训练模型
        model = SentenceTransformer(LOCAL_MODEL_PATH)
        print("句向量模型加载完成")
        return model
    except ImportError:
        print("警告：sentence-transformers 未安装，将仅使用关键词匹配")
        print("可以使用命令 'pip install sentence-transformers' 安装")
        return None


def precompute_tool_embeddings():
    """预计算所有工具描述的嵌入向量"""
    global _tool_embeddings
    
    model = get_embedding_model()
    if not model:
        return
    
    tool_descriptions = load_tool_descriptions()
    
    print("正在预计算工具描述的嵌入向量...")
    for name, info in tool_descriptions.items():
        # 合并详细描述和基本描述以获得更丰富的内容
        description = f"{info.get('description', '')} {info.get('detailed_description', '')}"
        _tool_embeddings[name] = model.encode(description)
    print(f"已完成 {len(_tool_embeddings)} 个工具的嵌入向量预计算")


def cosine_similarity(a, b):
    """计算两个向量之间的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@mcp.tool()
def query_tool(tool_name: str) -> Dict[str, Any]:
    """查询工具的详细信息。在使用任何注册工具前均需要使用此工具进行工具使用说明查询。
    
    Args:
        tool_name: 要查询的工具名称
        
    Returns:
        包含工具详细信息的字典
    """
    # 加载工具描述
    tool_descriptions = load_tool_descriptions()
    
    # 检查工具是否存在
    if tool_name not in tool_descriptions:
        return {
            "success": False,
            "error": f"工具 '{tool_name}' 不存在",
            "available_tools": list(tool_descriptions.keys())
        }
    
    # 获取工具信息
    tool_info = tool_descriptions[tool_name]
    
    # 构建响应
    response = {
        "success": True,
        "tool_name": tool_name,
        "description": tool_info.get("description", ""),
        "parameters": tool_info.get("parameters", {})
    }
    
    # 添加详细信息（如果有）
    if "detailed_description" in tool_info:
        response["detailed_description"] = tool_info["detailed_description"]
    
    # 添加返回值信息（如果有）
    if "returns" in tool_info:
        response["returns"] = tool_info["returns"]
    
    # 添加示例（如果有）
    if "examples" in tool_info:
        response["examples"] = tool_info["examples"][:3]  # 最多3个示例
    
    return response


@mcp.tool()
def list_available_tools() -> Dict[str, Any]:
    列出系统中所有注册工具名称和基本描述，列出后需要配合query_tool工具查询具体工具的使用说明。
    
    Returns:
        包含所有可用工具的基本信息
    """
    tool_descriptions = load_tool_descriptions()
    
    tool_list = {}
    for name, info in tool_descriptions.items():
        tool_list[name] = {
            "description": info.get("description", "")
        }
    
    return {
        "success": True,
        "available_tools": tool_list
    }


@mcp.tool()
def search_tools_by_task(task_description: str) -> Dict[str, Any]:
    """根据任务描述智能搜索相关工具。
    
    使用先进的嵌入向量搜索相结合的混合方法，
    基于语义理解和关键词出现来找到最相关的工具。嵌入向量搜索使用多语言句向量模型，
    能够理解任务的语义含义，即使没有直接的关键词匹配也能找到相关工具。
    （建议：优先使用，返回推荐工具，配合query_tool查询具体使用说明）
    
    Args:
        task_description: 任务描述文本，可以用自然语言描述需要完成的任务
        
    Returns:
        包含相关工具信息的字典，按相关性排序，包括匹配分数
    """
    tool_descriptions = load_tool_descriptions()
    
    # 结果集
    results = {}
    
    # 1. 关键词匹配方法
    keywords = task_description.lower().split()
    for name, info in tool_descriptions.items():
        # 合并基本描述和详细描述以提高匹配范围
        full_description = f"{info.get('description', '')} {info.get('detailed_description', '')}".lower()
        
        # 计算匹配度
        match_score = sum(1 for keyword in keywords if keyword in full_description)
        
        # 归一化分数到0-1之间
        total_keywords = len(keywords)
        normalized_score = match_score / total_keywords if total_keywords > 0 else 0
        
        results[name] = {
            "description": info.get("description", ""),
            "keyword_score": normalized_score,
            "combined_score": normalized_score  # 初始化为关键词分数
        }
    
    # 2. 嵌入向量搜索方法 - 只在这个函数被调用时才加载模型和预计算嵌入向量
    global _tool_embeddings
    
    # 获取模型（仅在需要时加载）
    model = get_embedding_model()
    if model:
        # 如果嵌入向量尚未预计算，则仅在这里进行预计算
        if not _tool_embeddings:
            precompute_tool_embeddings()
        
        # 获取查询的嵌入向量
        query_embedding = model.encode(task_description)
        
        # 计算每个工具的嵌入相似度
        for name in results.keys():
            if name in _tool_embeddings:
                embedding_similarity = cosine_similarity(query_embedding, _tool_embeddings[name])
                results[name]["embedding_score"] = float(embedding_similarity)
                
                # 综合分数：嵌入相似度 (权重0.8) + 关键词匹配 (权重0.2)
                results[name]["combined_score"] = 0.8 * embedding_similarity + 0.2 * results[name]["keyword_score"]
    
    # 按综合分数排序
    threshold = 0.3  # 相似度阈值
    filtered_results = {
        name: info for name, info in results.items() 
        if info["combined_score"] > threshold
    }
    
    sorted_matches = dict(sorted(
        filtered_results.items(),
        key=lambda x: x[1]["combined_score"],
        reverse=True
    )[:5])  # 最多返回5个
    
    return {
        "success": True,
        "matches": sorted_matches,
        "query": task_description
    }


if __name__ == "__main__":
    # 删除预加载逻辑，改为按需加载
    print(f"工具查询处理器启动中...")
    mcp.run() 
