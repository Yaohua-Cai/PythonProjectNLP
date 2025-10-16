# -*- coding: utf-8 -*-
# ================================
#  智谱 AI（ZhipuAI）GLM-4 简易 Agent Demo（Function Calling + 本地工具）
#  运行前准备：
#    1) 安装依赖：pip install zhipuai==2.* python-dotenv
#    2) 设置环境变量：export ZHIPUAI_API_KEY="你的apikey"  （Windows: set ZHIPUAI_API_KEY=你的apikey）
#    3) 可选：在项目根目录新建 docs/ 放入若干 .txt 文档，便于本地检索演示
#  用法：
#    python agent_demo.py  后按提示输入问题，输入 /exit 退出
#  说明：
#    - 该 Demo 展示“LLM + 工具调用（函数调用）+ 简单记忆与任务闭环”的最小可用样例
#    - 工具包含：计算器(calc)、当前时间(get_time)、读取文本(read_text)、关键字检索(search_local)
#    - 代码逐行中文注释，便于学习与二次扩展
# ================================

import os                          # 引入 os，用于读取环境变量（API Key）与文件路径操作
import json                        # 引入 json，用于在函数调用时解析/构造参数
import datetime as dt              # 引入 datetime 并简写为 dt，便于获取当前时间
import traceback                   # 引入 traceback，便于调试时打印完整异常栈
from typing import Dict, Any       # 引入类型标注，提升可读性（非必须）
from zhipuai import ZhipuAI        # 从 zhipuai SDK 导入 ZhipuAI 客户端

# ============ 基础配置 ============

MODEL_NAME = "glm-4"                                 # 指定使用的模型名称（可改为 glm-4-air / glm-4-flash / glm-4-long 等）
CLIENT = ZhipuAI(api_key="699fa03dbe994df4852607411ef48a00.lhkhZH3ycEnMj0TA")                    # 创建智谱客户端实例

# ============ Agent 的“系统提示” ============

SYSTEM_PROMPT = (
    "你是一个能够使用工具完成任务的智能体（Agent）。"
    "当你需要计算、读取本地文本、检索本地文档或获取当前时间时，请优先调用相应工具；"
    "如果没有足够证据，请提出澄清问题；"
    "输出应简洁可信，必要时附上来源（文件名/片段）。"
)  # 系统提示定义 Agent 的工作原则、工具使用规则与风格

# ============ 工具（函数）实现 ============

def tool_get_time() -> str:
    """返回当前本地时间的 ISO 字符串。"""
    # 获取当前时间，返回 ISO 格式，便于机器与人类同时阅读
    return dt.datetime.now().isoformat(timespec="seconds")


def _safe_eval_expr(expr: str) -> float:
    """一个非常简化/安全的表达式求值器，仅允许数字与 + - * / () 和小数点。"""
    # 去除空白字符，降低奇怪输入影响
    expr = "".join(ch for ch in expr if ch.strip() != "" or ch in " ")
    # 仅允许的字符集合，防止注入如 __import__、[] 等不安全内容
    allowed = set("0123456789.+-*/() ")
    if not set(expr) <= allowed:
        raise ValueError("表达式仅允许数字与 + - * / () 与小数点。")
    # 使用 Python 内置 eval 但处于极简安全模式：禁用 __builtins__，局部与全局均为空
    return eval(expr, {"__builtins__": {}}, {})


def tool_calc(expression: str) -> str:
    """计算简单的四则运算表达式。"""
    # 尝试安全求值，并将结果转为字符串返回
    value = _safe_eval_expr(expression)
    return str(value)


def tool_read_text(filename: str, max_chars: int = 1600) -> str:
    """读取 docs/ 目录下的 .txt 文本文件，返回前 max_chars 字符。"""
    # 统一 docs 目录，限制读取范围，避免越权访问上层目录
    base_dir = os.path.abspath("docs")
    # 仅保留纯文件名，防止路径穿越攻击（如 ../../etc/passwd）
    safe_name = os.path.basename(filename)
    # 强制限定扩展名为 .txt，避免读取二进制或敏感文件
    if not safe_name.lower().endswith(".txt"):
        raise ValueError("仅允许读取 .txt 文本文件。")
    # 拼接绝对路径并再次校验在 docs 目录内
    full_path = os.path.abspath(os.path.join(base_dir, safe_name))
    if not full_path.startswith(base_dir):
        raise PermissionError("非法路径访问。")
    # 判断文件是否存在
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"文件不存在：{safe_name}")
    # 读取文件内容，并限制输出长度，防止超长内容刷屏与成本上升
    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read(max_chars)
    # 返回带来源标识的片段，便于 Agent 在回答中引用
    return f"[来源: {safe_name}]\n{content}"


def tool_search_local(query: str, max_hits: int = 3, max_chars_per_hit: int = 400) -> str:
    """在 docs/ 目录下进行极简关键字检索，返回若干命中片段。"""
    # 约定检索目录
    base_dir = os.path.abspath("docs")
    if not os.path.isdir(base_dir):
        return "未找到 docs/ 目录，无法进行本地检索。"
    # 大小写不敏感检索
    q = query.strip().lower()
    hits = []  # 命中结果收集列表
    # 遍历 docs/ 下的 .txt 文件
    for name in os.listdir(base_dir):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(base_dir, name)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        # 简单匹配：若查询词出现，则截取其附近片段
        low = text.lower()
        idx = low.find(q)
        if idx != -1:
            start = max(0, idx - max_chars_per_hit // 2)
            end = min(len(text), idx + max_chars_per_hit // 2)
            snippet = text[start:end]
            hits.append(f"[来源: {name}]\n…{snippet}…")
        # 达到上限则停止
        if len(hits) >= max_hits:
            break
    # 若无命中则返回提示
    if not hits:
        return f"在本地文档中未检索到：{query}"
    # 将多条命中以分隔线拼接返回
    return "\n\n---\n\n".join(hits)


# ============ 工具（函数）清单：提供给模型的“工具元数据” ============

TOOLS_SPEC = [
    {   # 工具1：获取当前时间（无入参）
        "type": "function",                                           # 表示一个“函数型工具”
        "function": {
            "name": "get_time",                                       # 工具名称，需与本地实现同名
            "description": "获取当前本地时间（ISO 格式）。",              # 工具用途描述
            "parameters": {                                           # 入参 JSON Schema（无参数）
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {   # 工具2：计算器（允许简单四则运算）
        "type": "function",
        "function": {
            "name": "calc",
            "description": "计算简单的四则运算表达式（仅支持数字与 + - * / () 和小数点）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": { "type": "string", "description": "要计算的表达式，如 12*(3+4)/5" }
                },
                "required": ["expression"]
            }
        }
    },
    {   # 工具3：读取文本文件（docs/目录下）
        "type": "function",
        "function": {
            "name": "read_text",
            "description": "读取 docs/ 目录下的 .txt 文本文件，返回片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": { "type": "string", "description": "文件名（需以 .txt 结尾）" },
                    "max_chars": { "type": "integer", "description": "最大返回字符数，默认 1600", "minimum": 100 }
                },
                "required": ["filename"]
            }
        }
    },
    {   # 工具4：本地检索（极简关键字搜索）
        "type": "function",
        "function": {
            "name": "search_local",
            "description": "在 docs/ 目录下的 .txt 文档中进行简单关键字检索，返回若干命中片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "检索关键词" },
                    "max_hits": { "type": "integer", "description": "最大命中文档数，默认 3", "minimum": 1 }
                },
                "required": ["query"]
            }
        }
    }
]  # 以上列表会传给模型，使其能“发现”并调用这些工具


# ============ 工具名称到本地实现的映射（调度字典） ============

TOOL_IMPLS = {
    "get_time": lambda **kwargs: tool_get_time(),                                # 将工具名映射到对应函数（无参）
    "calc": lambda **kwargs: tool_calc(kwargs.get("expression", "")),            # 解析 expression 参数
    "read_text": lambda **kwargs: tool_read_text(                                # 解析 filename 与可选 max_chars
        kwargs.get("filename", ""), kwargs.get("max_chars", 1600)
    ),
    "search_local": lambda **kwargs: tool_search_local(                          # 解析 query 与可选 max_hits
        kwargs.get("query", ""), kwargs.get("max_hits", 3)
    )
}  # 该映射用于在收到“函数调用”请求时执行相应本地逻辑


# ============ 核心：一次对话轮的处理逻辑（带函数调用循环） ============

def run_agent_round(history: list, user_input: str) -> str:
    """
    进行单轮对话（可能包含一次或多次工具调用），返回模型最后的“自然语言答复”。
    :param history: 历史消息列表（含 system / user / assistant / tool）
    :param user_input: 本轮用户输入
    :return: 模型最终回答文本
    """
    # 步1：把本轮用户消息加入上下文
    history.append({"role": "user", "content": user_input})

    # 步2：首次请求模型，允许函数调用（tools）
    try:
        resp = CLIENT.chat.completions.create(                     # 调用智谱“聊天补全”接口
            model=MODEL_NAME,                                      # 使用的模型
            messages=history,                                      # 历史消息 + 当前用户消息
            tools=TOOLS_SPEC,                                      # 暴露“工具元数据”，启用函数调用能力
            temperature=0.3,                                       # 温度偏低以增强确定性
            top_p=0.9                                              # 可按需调整多样性
        )
    except Exception:
        # 若请求异常，打印错误并抛出，以便快速发现网络/鉴权/参数问题
        traceback.print_exc()
        raise

    # 步3：解析模型返回，判断是否提出“工具调用”请求
    msg = resp.choices[0].message                                  # 按 OpenAI 风格，取第一条候选的消息对象
    tool_calls = getattr(msg, "tool_calls", None) or getattr(msg, "function_call", None)
    # 兼容两种返回格式：
    #  - tool_calls：OpenAI 新版（列表，多工具并行）
    #  - function_call：旧版（单工具）

    # 如果模型没有调用工具，则直接返回答复并把消息写入历史
    if not tool_calls:
        assistant_text = msg.content or ""
        history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    # 规范化工具调用为列表形式，便于统一处理
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]

    # 步4：逐个执行工具调用，并把“工具结果”回填给模型
    for tc in tool_calls:
        # 不同 SDK 结构差异：尽量兼容 name 与 arguments 的获取方式
        if hasattr(tc, "function"):                                 # 新式结构：tc.function.name / tc.function.arguments
            name = getattr(tc.function, "name", "")
            args_json = getattr(tc.function, "arguments", "{}")
            call_id = getattr(tc, "id", "")                         # tool_call_id（若有）
        else:                                                       # 旧式结构：tc["name"] / tc["arguments"]
            name = tc.get("name", "")
            args_json = tc.get("arguments", "{}")
            call_id = tc.get("id", "")

        # 将 JSON 字符串解析为字典，若解析失败则当作空参数
        try:
            args = json.loads(args_json) if isinstance(args_json, str) else (args_json or {})
        except Exception:
            args = {}
        # 根据工具名称，从映射表中找到实现并执行；异常需捕获为字符串返回给模型
        try:
            impl = TOOL_IMPLS.get(name)
            if impl is None:
                result = f"工具未实现：{name}"
            else:
                result = impl(**args)
        except Exception as e:
            result = f"工具执行异常：{type(e).__name__}: {e}"

        # 将“工具运行结果”作为一条 role=tool 的消息追加到历史，带上 tool_call_id 以便模型关联
        history.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": str(result)
        })

    # 步5：把工具结果给到模型，让其综合生成最终答复（再次调用）
    resp2 = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=history,
        temperature=0.3,
        top_p=0.9
    )
    final_text = resp2.choices[0].message.content or ""            # 取最终自然语言回答
    history.append({"role": "assistant", "content": final_text})   # 写回历史，保持上下文连续
    return final_text                                               # 返回给调用者（终端打印）


# ============ 命令行交互入口（含“短期记忆”） ============

def main():
    """命令行循环：连续对话，直到用户输入 /exit 退出。"""
    # 初始化对话历史，先放入 system 提示，强化 Agent 的角色与工具使用策略
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    # 打印简单欢迎信息与提示
    print(" 简易Agent 已就绪，输入问题开始对话；输入 /exit 退出。")
    # 进入交互循环
    while True:
        try:
            # 读取用户输入
            user_input = input("\n你：").strip()
            # 若输入为空，则继续下一轮
            if not user_input:
                continue
            # 输入 /exit 则退出程序
            if user_input.lower() in ("/exit", "exit", "quit", "/quit"):
                print("再见！")
                break
            # 调用核心函数执行一轮 Agent 推理（包含可能的函数调用）
            answer = run_agent_round(history, user_input)
            # 在终端打印模型的最终答复
            print(f"\nAgent：{answer}")
        except KeyboardInterrupt:
            # 捕获 Ctrl+C，友好退出
            print("\n已中断，退出。")
            break
        except Exception as e:
            # 捕获其他异常并打印，便于排障
            print("\n⚠发生错误：", e)
            traceback.print_exc()
            # 为避免对话状态错乱，这里不退出，可按需清空 history 重来
            continue


# ============ 程序入口保护（仅作为脚本运行时执行 main） ============

if __name__ == "__main__":  # 当直接运行该文件时执行 main()，被其他文件 import 时不执行
    main()                  # 启动命令行交互
