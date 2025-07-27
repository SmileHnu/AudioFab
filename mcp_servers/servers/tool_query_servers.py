import json
import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from functools import lru_cache

from mcp.server.fastmcp import FastMCP

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# 创建MCP服务器
mcp = FastMCP("Tool Query Processor")


# NOTE: 使用相对路径自动定位到项目根目录，避免硬编码绝对路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # mcp_servers 目录
# 工具描述文件路径 （all:local + API）

TOOL_DESCRIPTIONS_PATH: Path = PROJECT_ROOT  / "tool_descriptions.json"

# 工具描述文件路径 （API:ONLY API）
# TOOL_DESCRIPTIONS_PATH: Path = PROJECT_ROOT / "tool_decs" / "tool_descriptions_api.json"


# 全局变量，用于存储句向量模型和工具描述的嵌入向量

_model = None
_tool_embeddings = {}

# 本地预训练模型路径
LOCAL_MODEL_PATH = "/home/chengz/LAMs/pre_train_models/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"

# 全局变量，用于存储Embedding和Reranker模型
_EMBEDDING_MODEL = None
_RERANKER_TOKENIZER = None
_RERANKER_MODEL = None

# 全局变量，用于存储工具描述，避免重复加载
_TOOL_DESCRIPTIONS = None

LOCAL_Embedding_PATH = "/home/chengz/LAMs/Eva_LLM_CK/models--Qwen--Qwen3-Embedding-0.6B"
LOCAL_Reranker_PATH = "/home/chengz/LAMs/Eva_LLM_CK/models--Qwen--Qwen3-Reranker-0.6B"


def load_tool_descriptions() -> Dict[str, Any]:
    """加载工具描述文件"""
    global _TOOL_DESCRIPTIONS
    
    # 如果已经加载过，直接返回缓存的结果
    if _TOOL_DESCRIPTIONS is not None:
        return _TOOL_DESCRIPTIONS
    
    try:
        if not TOOL_DESCRIPTIONS_PATH.exists():
            _TOOL_DESCRIPTIONS = {}
            return {}
        
        print(f"首次加载工具描述文件: {TOOL_DESCRIPTIONS_PATH}")
        with open(TOOL_DESCRIPTIONS_PATH, "r", encoding="utf-8") as f:
            _TOOL_DESCRIPTIONS = json.load(f)
        print(f"成功加载 {len(_TOOL_DESCRIPTIONS)} 个工具描述")
        return _TOOL_DESCRIPTIONS
    except Exception as e:
        print(f"加载工具描述出错: {e}")
        _TOOL_DESCRIPTIONS = {}
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
    """
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




#### 新搜索工具V2.0 ####
'''
根据Embeding 模型对Query 查询和tools json中工具描述进行语义相似度计算，然后使用Reranker 模型对结果进行重排序，最后返回前top k个结果。

'''

@mcp.tool()
def retrieve_tools(query: str, top_k: int = 3,initial_k: int = 10) -> Dict[str, Any]:
    """
    Use the Embedding-model and Reranker-model for two-stage retrieval to find the most relevant tools for a given query.

    This tool adopts an advanced two-stage retrieval strategy:
    1. First, the Embedding-model encodes the query and tool descriptions semantically, selecting initial candidate tools based on cosine similarity.
    2. Then, the Reranker-model performs a refined ranking of the candidate tools and returns the most relevant ones.

    The more detailed the query, the more accurate the matching results!

    Args:
        query: A detailed query text describing the task to be accomplished
        top_k: Number of most relevant tools to return, default is 3
        initial_k: Number of initial candidates, default is 10
    Returns:
        A dictionary containing information on the most relevant tools, sorted by relevance
    """
    # ---------------- 参数检查 ----------------
    print(f"query: {query}")
    if not query:
        return {
            "success": False,
            "error": "Query is a required parameter. Please provide a detailed description of the subtask!",
            "results": []
        }
    
    if not isinstance(query, str):
        query = str(query)
    
    initial_k = initial_k  # 初始候选数量
    
    # ---------------- 依赖导入 ----------------
    global _EMBEDDING_MODEL, _RERANKER_TOKENIZER, _RERANKER_MODEL
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    
    # 1) 加载/缓存 Embedding 模型
    if _EMBEDDING_MODEL is None:
        try:
            print("[Retriever] Loading Embedding-model ...")
            _EMBEDDING_MODEL = SentenceTransformer(LOCAL_Embedding_PATH)
            _EMBEDDING_MODEL.to(device)
            _EMBEDDING_MODEL.eval()
        except Exception as e:
            print(f"[Retriever] 加载嵌入模型失败: {e}")
            return {
                "success": False,
                "error": "嵌入模型加载失败",
                "results": []
            }
    
    # 2) 加载/缓存 Reranker 模型
    if _RERANKER_MODEL is None:
        try:
            print("[Retriever] Loading Reranker-model ...")
            _RERANKER_TOKENIZER = AutoTokenizer.from_pretrained(
                LOCAL_Reranker_PATH, padding_side="left")
            _RERANKER_MODEL = AutoModelForCausalLM.from_pretrained(
                LOCAL_Reranker_PATH).to(device).eval()
        except Exception as e:
            print(f"[Retriever] 加载重排模型失败: {e}")
            return {
                "success": False,
                "error": "重排模型加载失败",
                "results": []
            }
    
    # ---------------- 加载工具描述 ----------------
    tool_descriptions = load_tool_descriptions()
    if not tool_descriptions:
        return {
            "success": False,
            "error": "工具描述文件为空或无法加载",
            "results": []
        }
    
    # ---------------- 嵌入函数 ----------------
    def _embed(texts: List[str], is_query: bool = False, batch_size: int = 32):
        if is_query:
            return _EMBEDDING_MODEL.encode(texts, prompt_name="query", batch_size=batch_size, convert_to_numpy=True)
        return _EMBEDDING_MODEL.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    
    # 计算 query embedding
    query_vec = _embed([query], is_query=True)[0].reshape(1, -1)
    
    # ---------------- 直接使用工具描述进行嵌入和排序 ----------------
    # 保存工具名称和对应的嵌入向量
    tool_names = []
    tool_vecs = []
    
    # 简单的嵌入缓存实现
    # 使用全局变量作为嵌入缓存
    global _TOOL_EMBEDDING_CACHE
    if not globals().get('_TOOL_EMBEDDING_CACHE'):
        _TOOL_EMBEDDING_CACHE = {}
    
    for name, tool in tool_descriptions.items():
        # 合并description和detailed_description
        combined_desc = tool.get('description', '')
        if 'detailed_description' in tool:
            combined_desc += " " + tool['detailed_description']
        
        # 计算嵌入向量
        cache_key = name
        vec = _TOOL_EMBEDDING_CACHE.get(cache_key)
        if vec is None:
            vec = _embed([combined_desc])[0]
            _TOOL_EMBEDDING_CACHE[cache_key] = vec
        
        tool_names.append(name)
        tool_vecs.append(vec)
    
    if not tool_names:
        return {
            "success": False,
            "error": "No available tool!",
            "results": []
        }
    
    tool_vecs_np = np.vstack(tool_vecs)
    
    # ---------------- 粗排 ----------------
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
    sims = sk_cosine_similarity(query_vec, tool_vecs_np)[0]
    initial_k = min(initial_k, len(sims))
    top_order = np.argsort(sims)[-initial_k:][::-1]
    
    # 获取初步候选工具
    cand_tools = []
    cand_descs = []
    for idx in top_order:
        name = tool_names[idx]
        tool = tool_descriptions[name]
        
        # 合并description和detailed_description
        combined_desc = tool.get('description', '')
        if 'detailed_description' in tool:
            combined_desc += " " + tool['detailed_description']
        
        cand_tools.append((name, tool))
        cand_descs.append(combined_desc)
    
    # ---------------- 精排 ----------------
    def _format_pair(q: str, d: str, instruction: str | None = None) -> str:
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}"
    
    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
        "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = _RERANKER_TOKENIZER.encode(prefix, add_special_tokens=False)
    suffix_tokens = _RERANKER_TOKENIZER.encode(suffix, add_special_tokens=False)
    token_true_id = _RERANKER_TOKENIZER.convert_tokens_to_ids("yes")
    token_false_id = _RERANKER_TOKENIZER.convert_tokens_to_ids("no")
    max_length = 8192
    
    def _pack_inputs(pairs: List[str]):
        enc = _RERANKER_TOKENIZER(
            pairs, padding=False, truncation='longest_first', return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        for i, ids in enumerate(enc['input_ids']):
            enc['input_ids'][i] = prefix_tokens + ids + suffix_tokens
        enc = _RERANKER_TOKENIZER.pad(enc, padding=True, return_tensors='pt', max_length=max_length)
        return {k: v.to(device) for k, v in enc.items()}
    
    @torch.no_grad()
    def _compute_scores(pairs: List[str]):
        inputs = _pack_inputs(pairs)
        logits = _RERANKER_MODEL(**inputs).logits[:, -1, :]
        true_vec = logits[:, token_true_id]
        false_vec = logits[:, token_false_id]
        stacked = torch.stack([false_vec, true_vec], dim=1)
        scores = torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp().cpu().tolist()
        return scores
    
    pairs = [_format_pair(query, d) for d in cand_descs]
    scores = _compute_scores(pairs)
    ranked = sorted(zip(scores, cand_tools), key=lambda x: x[0], reverse=True)[:top_k]
    
    # ----------- 构建结果 -------------
    results = []
    for sc, (name, tool) in ranked:
        # 合并description和detailed_description
        combined_desc = tool.get('description', '')
        if 'detailed_description' in tool:
            combined_desc += " " + tool['detailed_description']
        
        result_item = {
            "name": name,
            "description": combined_desc,
            "score": float(sc)
        }
        
        # 添加参数信息（如果有）
        if "parameters" in tool and tool["parameters"]:
            result_item["parameters"] = tool["parameters"]
        
        # 添加示例（如果有）
        if "examples" in tool and tool["examples"]:
            result_item["examples"] = tool["examples"]
            
        results.append(result_item)
    
    return {
        "success": True,
        "results": results
    }



if __name__ == "__main__":
    # 在服务启动时预加载工具描述
    print(f"工具查询处理器启动中...")
    print(f"预加载工具描述文件...")
    tool_descriptions = load_tool_descriptions()
    print(f"已加载 {len(tool_descriptions)} 个工具描述")
    
    # 初始化全局嵌入缓存
    _TOOL_EMBEDDING_CACHE = {}
    
    mcp.run() 