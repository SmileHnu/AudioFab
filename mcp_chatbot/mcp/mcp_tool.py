from typing import Any

'''
作用：将工具的关键信息（名称、描述、参数及其说明）格式化为适合 LLM 理解的字符串。
步骤：
检查 input_schema 是否有 "properties" 字段（即参数定义）。
遍历每个参数，提取参数名和参数描述。
如果参数是必填项（在 "required" 列表中），则在描述后加上 (required)。
最终拼接成如下格式的字符串：

Tool: audio_enhance
Description: Enhance audio quality
Arguments:
- audio_data: Base64 encoded audio data (required)
- noise_level: Level of noise reduction

'''



class MCPTool:
    """Represents a MCP tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""
