import os
import json
from pathlib import Path
from typing import Union

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("万能读写文件工具")

@mcp.tool()
def read_file(directory_path: str, file_type: str = "md") -> Union[str, dict]:
    """
    读取指定类型的所有文件内容，支持md, txt, json文件。json文件以dict返回，其他文件合并字符串返回。
    Args:
        directory_path: 文件所在目录
        file_type: 文件类型(md, txt, json)
    Returns:
        所有该目录下对应类型文件的内容合集 或 error 信息
    """
    if file_type not in ("md", "txt", "json"):
        return {"error": f"不支持的文件类型: {file_type}"}

    try:
        suffix = "." + file_type
        file_paths = list(Path(directory_path).glob(f"*{suffix}"))
        if not file_paths:
            return {"error": f"{directory_path} 中没有 {file_type} 文件"}

        if file_type == "json":
            datas = []
            for fp in file_paths:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        datas.append(json.load(f))
                except Exception as e:
                    datas.append({"error": f"文件{fp}无法解析: {e}"})
            return datas
        else:
            contents = []
            for fp in file_paths:
                with open(fp, "r", encoding="utf-8") as f:
                    contents.append(f"########## [{fp.name}] ##########\n" + f.read())
            return "\n\n".join(contents)
    except Exception as e:
        return {"error": f"读取文件时出错: {e}"}


@mcp.tool()
def write_file(directory_path: str, filename: str, content: str, overwrite: bool = False) -> str:
    """
    写入（或新建）指定类型文件，支持md, txt, json。如果json自动解析为json对象写入。
    Args:
        directory_path: 文件夹路径
        filename: 文件名（须带后缀或自动补全）
        content: 文件内容（写入json时必须为可解析的json格式字符串）
        overwrite: 如为True则覆盖原文件
    Returns:
        执行结果说明
    """
    EXT_ALLOWED = [".md", ".txt", ".json"]
    # 自动补后缀
    ext = os.path.splitext(filename)[1].lower()
    if ext not in EXT_ALLOWED:
        if filename.endswith("."):
            filename = filename[:-1]
        filename += ".md"
        ext = ".md"

    file_path = Path(directory_path) / filename
    os.makedirs(file_path.parent, exist_ok=True)

    if file_path.exists() and not overwrite:
        return f"Error: 文件 {file_path} 已存在。（如需覆盖请设置 overwrite=True）"

    try:
        if ext == ".json":
            try:
                json_obj = content if isinstance(content, dict) else json.loads(content)
            except Exception as je:
                return f"Error: json内容解析异常: {je}"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, ensure_ascii=False, indent=2)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        return f"Success: 文件写入 {file_path}"
    except Exception as e:
        return f"写文件出错: {e}"


@mcp.tool()
def modify_file(file_path: str, new_content: str) -> str:
    """
    修改（覆盖）单个已存在的.md/.txt/.json文件。json支持自动序列化。
    Args:
        file_path: 要修改的文件完整路径
        new_content: 新内容（json类型自动处理）
    Returns:
        执行结果说明
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in [".md", ".txt", ".json"]:
        return f"只支持md、txt、json类型文件修改，当前: {ext}"
    try:
        if ext == ".json":
            try:
                new_obj = new_content if isinstance(new_content, dict) else json.loads(new_content)
            except Exception as je:
                return f"Error: 新json内容解析失败: {je}"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(new_obj, f, ensure_ascii=False, indent=2)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        return f"Success: 文件已修改 {file_path}"
    except Exception as e:
        return f"修改文件时出错: {e}"


if __name__ == "__main__":
    mcp.run()
