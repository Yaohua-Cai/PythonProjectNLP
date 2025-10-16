# -*- coding: utf-8 -*-
# ===============================================================
#  通用 MCP 服务器（Python，官方 mcp Python SDK）+ 智谱 GLM 工具示例
#  功能：同时演示 MCP 的三大能力 —— Tools / Resources / Prompts
#        并提供一个调用【智谱AI】GLM 模型的工具（需环境变量 ZHIPUAI_API_KEY）
#
#  依赖安装（任选其一）：
#    pip install "mcp[cli]" zhipuai               # 官方 MCP Python SDK + 智谱 SDK
#    # 或者使用 uv：
#    # uv add "mcp[cli]" ; uv add zhipuai
#
#  目录准备：
#    项目根目录下新建 docs/ 并放若干 .txt 文件，供资源（resources）读取演示
#
#  运行与连接（两种常见方式）：
#    方式A（建议开发调试）：mcp dev mcp.py
#    方式B（被 MCP 客户端/宿主以 stdio 子进程方式拉起）：python mcp.py
#    * 大多数 MCP 客户端（如 Claude Desktop / MCP Inspector）会“自己拉起”本进程（stdio）
#      若手动运行本脚本，会进入“等待宿主连接”的状态
# ===============================================================

import os                                              # 读取环境变量、拼接路径
from typing import List                                # 类型注解：列表
from datetime import datetime                          # 获取当前时间（工具示例）
from mcp.server.fastmcp import FastMCP                 # 官方 SDK 的“快速服务器”封装
from mcp.server.fastmcp import Context                 # （可选）在工具里拿到上下文
from mcp.server.session import ServerSession           # （可选）上下文中的会话类型

# —— 可选：尝试导入智谱 SDK；若未安装或无密钥，相关工具会返回友好提示 ——
try:
    from zhipuai import ZhipuAI                        # 智谱AI 官方 Python SDK
except Exception:                                      # 导入失败（未安装）
    ZhipuAI = None                                     # 占位，避免模块级报错

# ============== 基本配置 ==============

SERVER_NAME = "Zhipu-MCP-Demo"                         # 服务器对外显示的名称
DOCS_DIR = os.path.abspath("docs")                     # 本地资源目录（仅暴露 .txt）

# ============== 创建 MCP 服务器实例 ==============

mcp = FastMCP(SERVER_NAME)                             # FastMCP：封装了 MCP 协议与路由

# ============== 定义 Tools（可被 LLM 调用的动作） ==============

@mcp.tool()                                            # 用装饰器将函数注册为 MCP Tool
def get_time() -> str:                                 # 工具：返回当前本地时间
    """获取当前本地时间（ISO8601）。"""                    # 工具描述（客户端可见）
    return datetime.now().isoformat(timespec="seconds")# 实际返回：形如 2025-09-17T10:11:12

@mcp.tool()                                            # 注册第二个工具：安全四则运算
def calc(expression: str) -> str:                      # 入参由 MCP 客户端/LLM 传入
    """计算四则运算表达式（仅允许 0-9.+-*/() 空格）。"""     # 约束：仅安全字符
    allowed = set("0123456789.+-*/() ")               # 允许字符白名单
    if not set(expression) <= allowed:                 # 若包含非法字符
        return "表达式仅允许数字与 + - * / () 与空格/小数点。" # 直接返回错误提示
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # 禁用内建，执行表达式
    except Exception as e:                             # 计算失败（括号不配等）
        return f"计算失败：{type(e).__name__}: {e}"       # 返回错误文本，便于模型理解

@mcp.tool()                                            # 注册第三个工具：调用智谱 GLM 摘要
def summarize_with_glm(text: str, max_words: int = 120) -> str:
    """使用智谱 GLM 对文本做要点总结（需 ZHIPUAI_API_KEY）。"""  # 工具说明
    client = ZhipuAI(api_key="699fa03dbe994df4852607411ef48a00.lhkhZH3ycEnMj0TA")                # 构造智谱客户端
    # 构造消息：系统提示限定输出风格与长度；用户消息放原文
    messages = [
        {"role": "system",
         "content": f"你是摘要助手。请在 {max_words} 字以内输出中文要点，使用短句与项目符号。"},
        {"role": "user", "content": text}
    ]
    # 调用聊天补全接口（可根据需要替换为 glm-4-air/flash/long 等）
    resp = client.chat.completions.create(
        model="glm-4",                                  # 模型名称
        messages=messages,                              # 对话消息
        temperature=0.2,                                # 低温度提升稳定性
        top_p=0.9                                       # 多样性阈值
    )
    return (resp.choices[0].message.content or "").strip()  # 返回模型文本

@mcp.tool()                                            # 演示：带进度反馈的工具（可观察日志流）
async def long_task(                                   # 异步工具可在执行中上报进度
    task_name: str = "demo",                           # 任务名参数（可选）
    steps: int = 5,                                    # 步数（演示用）
    ctx: Context[ServerSession, None] = None           # 注入 FastMCP 上下文（可选）
) -> str:
    """演示长任务：逐步上报进度（MCP 客户端可展示进度/日志）。"""   # 工具说明
    if ctx:                                            # 若拿到了上下文
        await ctx.info(f"开始任务：{task_name}")         # 发送 info 日志
    for i in range(steps):                             # 模拟多步执行
        if ctx:                                        # 每一步更新进度
            await ctx.report_progress(
                progress=(i + 1) / steps,             # 当前进度（0~1）
                total=1.0,                            # 总量（固定 1）
                message=f"Step {i + 1}/{steps}"       # 进度消息
            )
    return f"任务完成：{task_name}（共 {steps} 步）"          # 返回最终结果文本

# ============== 定义 Resources（只读资源，像 GET） ==============

@mcp.resource("docs://{name}")                         # 注册资源 URI 模板：docs://文件名
def read_local_txt(name: str) -> str:                  # 根据 {name} 读取 docs/ 下 .txt
    """读取本地 docs/ 目录下的 .txt 文件内容（资源只读）。"""   # 资源说明
    safe = os.path.basename(name)                      # 规避路径穿越（仅取文件名）
    if not safe.lower().endswith(".txt"):              # 仅允许 .txt
        return "仅允许读取 .txt 文本资源。"
    path = os.path.join(DOCS_DIR, safe)                # 拼接完整路径
    if not os.path.exists(path):                       # 文件必须存在
        return f"文件不存在：{safe}"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:  # 打开文件
        return f.read()                                # 返回全文（客户端会放入模型上下文）

@mcp.resource("docs-index://list")                     # 再注册一个“索引资源”
def list_local_txt() -> List[str]:                     # 用于列举 docs/ 下可读取的 .txt
    """列出 docs/ 目录下所有 .txt 文件名（用于浏览与选择）。"""   # 资源说明
    if not os.path.isdir(DOCS_DIR):                    # 若目录不存在
        return []                                      # 返回空列表
    return sorted(                                     # 返回排序后的文件名列表
        [n for n in os.listdir(DOCS_DIR) if n.lower().endswith(".txt")]
    )

# ============== 定义 Prompts（可复用的提示模板） ==============

@mcp.prompt()                                          # 注册一个“审稿”提示模板
def critic(title: str = "未命名") -> str:               # 可带参数（客户端在 get_prompt 时填入）
    """审稿提示：指出逻辑漏洞、证据不足与潜在风险。"""          # 模板说明
    return (
        "你是严格的审稿人，请对下列文稿给出改进意见：\n"
        f"《{title}》\n"
        "要求：\n"
        "1) 列出主要问题（逻辑/证据/结构/可读性）；\n"
        "2) 给出可执行修改建议；\n"
        "3) 以要点列表输出。"
    )

@mcp.prompt()                                          # 注册一个“邮件摘要”提示模板
def email_summarizer() -> str:                         # 无参数模板
    """邮件摘要提示：三句话总结 + 行动项（Assignee/DDL）。"""     # 模板说明
    return (
        "请将以下邮件内容：\n"
        "1) 用三句话概括要点；\n"
        "2) 提取行动项（Assignee/DDL/阻塞点）。"
    )

# ============== 服务器启动（默认 stdio；由 MCP 客户端拉起） ==============

if __name__ == "__main__":                             # 被直接执行时才进入启动逻辑
    # FastMCP.run()：默认使用 stdio 传输；适配 Claude Desktop / MCP Inspector 等宿主
    # 如需 HTTP/SSE 等远程形态，可参考官方文档切换 transport 参数（示例：mcp.run(transport="streamable-http")）
    mcp.run()                                          # 启动 MCP 服务器（阻塞等待宿主连接）
