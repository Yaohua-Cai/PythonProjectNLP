# -*- coding: utf-8 -*-
# ==============================================================
#  智谱AI GLM 系列 + 工具调用 的最小可用 Agent 示例
#  目标：用“可运行”的方式演示 Agent 的四大核心组成部分：
#        1) Planner（规划器） 2) Executor（执行器/工具调用）
#        3) Memory（记忆：短期/长期） 4) Tools（工具/环境接口）
#
#  先决条件：
#    pip install zhipuai==2.* python-dotenv
#    export ZHIPUAI_API_KEY="你的apikey"   # Windows: set ZHIPUAI_API_KEY=xxx
#
#  运行方式：
#    python agent_core_demo.py
#    在项目根目录下可创建 docs/ 目录，放入 .txt 文档用于本地检索/读取演示
# ==============================================================

import os                                  # 读取环境变量、构造路径
import json                                # 解析/输出 JSON（用于函数调用参数与规划结果）
import time                                # 简单的时间控制（演示延迟/节流）
import datetime as dt                      # 获取当前时间（工具示例）
import traceback                           # 打印异常堆栈便于排错
from typing import List, Dict, Any         # 类型注解，增强可读性（非必须）
from zhipuai import ZhipuAI                # 智谱AI 官方 Python SDK 客户端

# ==========================
#  全局配置
# ==========================

MODEL_NAME = "glm-4"                                  # 选择模型（可替换为 glm-4-air / glm-4-flash / glm-4-long 等）
CLIENT = ZhipuAI(api_key="699fa03dbe994df4852607411ef48a00.lhkhZH3ycEnMj0TA")                       # 创建智谱AI客户端实例

# ==========================
#  Agent 的“系统提示词”
# ==========================

SYSTEM_PROMPT = (
    "你是一个可使用工具执行任务的智能体（Agent）。"
    "请先规划步骤，再逐步执行；遇到计算/时间/读取本地文本/检索本地文档时，优先调用工具。"
    "回答要求：简洁、可验证；引用本地内容时请标注来源文件名。"
)  # 为模型提供稳定的角色与行为规范

# ==========================
#  Memory（记忆）实现
# ==========================

class ShortTermMemory:
    """短期记忆：维护本次会话的消息历史，控制长度避免过长上下文。"""
    def __init__(self, max_messages: int = 30):                 # 初始化，设定最多保留的消息条数
        self.history: List[Dict[str, Any]] = []                 # 用列表存放消息字典（role/content/...）
        self.max_messages = max_messages                        # 记录上限

    def add(self, role: str, content: str, **kwargs):           # 追加一条消息到历史
        msg = {"role": role, "content": content}                # 基础字段：角色与内容
        msg.update(kwargs)                                      # 允许携带额外字段（如 tool_call_id、name）
        self.history.append(msg)                                # 实际存储
        if len(self.history) > self.max_messages:               # 若超过上限则裁剪旧消息
            self.history = self.history[-self.max_messages:]    # 保留最新的若干条

    def get(self) -> List[Dict[str, Any]]:                      # 读取历史消息
        return list(self.history)                               # 返回一份拷贝，避免外部意外修改


class LongTermMemory:
    """长期记忆：简单的键值/事实存储（JSON文件），支持追加和按关键词检索。"""
    def __init__(self, path: str = "ltm.json"):                 # 指定持久化文件路径
        self.path = path                                        # 记录路径
        if not os.path.exists(self.path):                       # 若文件不存在则初始化空结构
            with open(self.path, "w", encoding="utf-8") as f:   # 打开文件用于写入
                json.dump({"facts": []}, f, ensure_ascii=False, indent=2)  # 写入初始 JSON

    def _load(self) -> Dict[str, Any]:                          # 内部方法：加载 JSON 内容
        with open(self.path, "r", encoding="utf-8") as f:       # 打开文件
            return json.load(f)                                 # 返回解析后的字典

    def _save(self, data: Dict[str, Any]):                      # 内部方法：保存 JSON 内容
        with open(self.path, "w", encoding="utf-8") as f:       # 打开文件写入
            json.dump(data, f, ensure_ascii=False, indent=2)    # 漂亮格式输出

    def add_fact(self, fact: str, tags: List[str] = None):      # 对外方法：追加一条“长期记忆事实”
        data = self._load()                                     # 读取当前数据
        data["facts"].append({                                  # 追加结构化对象
            "fact": fact,                                       # 事实文本
            "tags": tags or [],                                 # 标签列表（可用于简单过滤）
            "time": dt.datetime.now().isoformat(timespec="seconds")  # 添加时间戳
        })
        self._save(data)                                        # 保存到磁盘

    def search(self, keyword: str, top_k: int = 3) -> List[str]:# 关键词检索，返回若干命中事实文本
        data = self._load()                                     # 读取数据
        hits = []                                               # 命中列表
        low_kw = keyword.lower()                                # 小写化关键词便于大小写不敏感匹配
        for item in data.get("facts", []):                      # 遍历事实
            if low_kw in item.get("fact", "").lower():          # 若包含关键词
                hits.append(f"- {item['fact']} （{item['time']}）") # 格式化结果
            if len(hits) >= top_k:                              # 命中达到上限则提前返回
                break
        return hits                                             # 返回命中列表


# ==========================
#  Tools（工具）实现
# ==========================

def tool_get_time() -> str:
    """返回当前本地时间（ISO 格式）。"""
    return dt.datetime.now().isoformat(timespec="seconds")      # 直接使用 datetime 生成 ISO 字符串

def _safe_eval_expr(expr: str) -> float:
    """极简安全计算器：仅允许数字与 + - * / () ."""
    expr = "".join(ch for ch in expr if ch.strip() != "" or ch in " ")  # 去除奇怪空白
    allowed = set("0123456789.+-*/() ")                       # 允许字符集合
    if not set(expr) <= allowed:                              # 若出现非法字符则报错
        raise ValueError("表达式仅允许数字与 + - * / () 与小数点。")
    return eval(expr, {"__builtins__": {}}, {})               # 禁用内建，防止注入

def tool_calc(expression: str) -> str:
    """计算四则运算表达式并返回字符串结果。"""
    return str(_safe_eval_expr(expression))                   # 计算后转字符串

def tool_read_text(filename: str, max_chars: int = 1200) -> str:
    """读取 docs/ 目录下 .txt 文件的前若干字符。"""
    base_dir = os.path.abspath("docs")                        # 限定基础目录
    safe_name = os.path.basename(filename)                    # 去路径，防止穿越
    if not safe_name.lower().endswith(".txt"):                # 仅允许 .txt 文件
        raise ValueError("仅允许读取 .txt 文本文件。")
    full = os.path.abspath(os.path.join(base_dir, safe_name)) # 拼出绝对路径
    if not full.startswith(base_dir):                         # 再次校验目录边界
        raise PermissionError("非法路径访问。")
    if not os.path.exists(full):                              # 检查文件存在性
        raise FileNotFoundError(f"文件不存在：{safe_name}")
    with open(full, "r", encoding="utf-8", errors="ignore") as f:  # 打开文件读取
        content = f.read(max_chars)                           # 读取限制长度
    return f"[来源: {safe_name}]\n{content}"                  # 返回带来源的片段

def tool_search_local(query: str, max_hits: int = 3, max_chars_per_hit: int = 400) -> str:
    """在 docs/ 目录进行极简关键字检索，返回若干命中片段。"""
    base_dir = os.path.abspath("docs")                        # 限定检索目录
    if not os.path.isdir(base_dir):                           # 若目录不存在直接提示
        return "未找到 docs/ 目录。"
    q = query.strip().lower()                                 # 关键词规范化
    hits = []                                                 # 命中列表
    for name in os.listdir(base_dir):                         # 遍历文件
        if not name.lower().endswith(".txt"):                 # 仅处理 .txt
            continue
        path = os.path.join(base_dir, name)                   # 拼路径
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:  # 读取文件
                text = f.read()                               # 读取全文（简化处理）
        except Exception:
            continue                                          # 读取异常则跳过
        low = text.lower()                                    # 小写化文本
        idx = low.find(q)                                     # 查找关键词
        if idx != -1:                                         # 若有命中
            start = max(0, idx - max_chars_per_hit // 2)      # 片段开始位置
            end = min(len(text), idx + max_chars_per_hit // 2)# 片段结束位置
            snippet = text[start:end]                         # 截取片段
            hits.append(f"[来源: {name}]\n…{snippet}…")       # 格式化命中
        if len(hits) >= max_hits:                             # 达到上限即停止
            break
    if not hits:                                              # 若无命中
        return f"未在本地文档中检索到：{query}"               # 返回提示
    return "\n\n---\n\n".join(hits)                           # 用分隔符拼接命中片段

# ==========================
#  工具元数据（供模型发现与调用）
# ==========================

TOOLS_SPEC = [
    {   # 工具：当前时间
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前本地时间（ISO 格式）。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {   # 工具：计算器
        "type": "function",
        "function": {
            "name": "calc",
            "description": "计算简单四则运算表达式（仅支持数字与 + - * / () .）。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "表达式，如 12*(3+4)/5"}},
                "required": ["expression"]
            }
        }
    },
    {   # 工具：读取文本
        "type": "function",
        "function": {
            "name": "read_text",
            "description": "读取 docs/ 目录下的 .txt 文本，返回片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "文件名（需以 .txt 结尾）"},
                    "max_chars": {"type": "integer", "description": "最大返回字符数，默认 1200", "minimum": 100}
                },
                "required": ["filename"]
            }
        }
    },
    {   # 工具：本地检索
        "type": "function",
        "function": {
            "name": "search_local",
            "description": "在 docs/ 下进行关键字检索，返回若干命中片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词"},
                    "max_hits": {"type": "integer", "description": "最大命中条数，默认 3", "minimum": 1}
                },
                "required": ["query"]
            }
        }
    }
]  # 以上结构让模型知道可用工具的名称、用途与参数Schema

# 映射“工具名 → 本地实现函数”（执行器将用到）
TOOL_IMPLS = {
    "get_time": lambda **kw: tool_get_time(),                          # 无参工具
    "calc": lambda **kw: tool_calc(kw.get("expression", "")),          # 解析 expression
    "read_text": lambda **kw: tool_read_text(kw.get("filename", ""), kw.get("max_chars", 1200)),  # 解析 filename/max_chars
    "search_local": lambda **kw: tool_search_local(kw.get("query", ""), kw.get("max_hits", 3))    # 解析 query/max_hits
}

# ==========================
#  Planner（规划器）实现
# ==========================

class Planner:
    """Planner：负责把用户目标转为可执行的步骤（粗粒度计划）。"""
    def __init__(self, client: ZhipuAI, model: str):               # 记录客户端与模型名
        self.client = client                                       # 存储智谱客户端
        self.model = model                                         # 存储使用的模型

    def plan(self, goal: str) -> List[Dict[str, Any]]:             # 输入用户目标，返回步骤列表
        prompt = (                                                 # 构造提示，要求以 JSON 数组返回步骤
            "请为以下目标生成 3-6 步的执行计划，JSON 数组格式输出：\n"
            "每步包含：step、intent、suggested_tool（可为none）、notes。\n"
            f"目标：{goal}\n"
            "仅输出 JSON，不要任何解释。"
        )
        # 调用聊天补全接口获取规划文本
        resp = self.client.chat.completions.create(                # 请求模型生成计划
            model=self.model,                                      # 指定模型
            messages=[{"role": "system", "content": "你是规划助手。"},  # 设定角色为规划助手
                      {"role": "user", "content": prompt}],        # 提供用户需求
            temperature=0.2                                        # 低温度，提升确定性
        )
        text = resp.choices[0].message.content or "[]"             # 取输出文本，容错缺省为 []
        try:
            plan = json.loads(text)                                # 尝试解析 JSON
            if isinstance(plan, list):                             # 确认是列表
                return plan                                        # 返回规划
        except Exception:
            pass                                                   # 若解析失败，走回退策略
        # 解析失败时构造一个保底的两步计划
        return [
            {"step": 1, "intent": "澄清需求", "suggested_tool": "none", "notes": "确认目标/约束/所需信息"},
            {"step": 2, "intent": "尝试回答", "suggested_tool": "search_local", "notes": "检索本地资料后整合作答"}
        ]  # 返回降级计划，保证后续流程可继续

# ==========================
#  Executor（执行器）实现（含工具调用循环）
# ==========================

class Agent:
    """Agent：聚合 Planner / Memory / Tools，执行任务闭环。"""
    def __init__(self, client: ZhipuAI, model: str):               # 初始化时注入客户端与模型
        self.client = client                                       # 存下客户端
        self.model = model                                         # 存下模型名
        self.planner = Planner(client, model)                       # 创建规划器
        self.stm = ShortTermMemory(max_messages=40)                 # 创建短期记忆（消息历史）
        self.ltm = LongTermMemory(path="ltm.json")                  # 创建长期记忆（简单 JSON 文件）

        # 初始化时在记忆里加入系统提示，保证角色稳定
        self.stm.add("system", SYSTEM_PROMPT)                       # 将系统提示加入消息历史

    def _append_tool_result(self, call_id: str, name: str, result: str):  # 内部：将工具结果写入历史
        self.stm.add("tool", result, tool_call_id=call_id, name=name)     # 记录工具调用结果（role=tool）

    def _call_model(self, messages: List[Dict[str, Any]], with_tools: bool = True):  # 内部：统一的模型调用
        # 构造请求参数，如果 with_tools=True 则附上 TOOLS_SPEC 允许函数调用
        params = dict(model=self.model, messages=messages, temperature=0.3, top_p=0.9)  # 基本参数
        if with_tools:                                                                  # 根据需要添加工具定义
            params["tools"] = TOOLS_SPEC                                                # 使模型“看见”可调用的工具
        # 正式发起请求
        return self.client.chat.completions.create(**params)                            # 返回模型响应对象

    def run(self, user_goal: str, max_tool_rounds: int = 4) -> str:  # 外部入口：执行一次完整任务
        # 第1步：让 Planner 产出粗粒度计划
        plan = self.planner.plan(user_goal)                            # 调用规划器
        self.ltm.add_fact(f"计划生成：{user_goal} -> 步数 {len(plan)}")  # 把“计划生成”事件写入长期记忆（用于审计/可追溯）
        # 把规划以系统可读的形式加入短期记忆，供模型参考
        self.stm.add("assistant", f"[规划草案]\n{json.dumps(plan, ensure_ascii=False, indent=2)}")  # 历史中记录计划概览
        # 同时把长期记忆中与本目标关键词相关的事实检索出来，压缩到一段上下文
        ltm_hits = self.ltm.search(keyword=user_goal, top_k=3)         # 基于关键词的简单检索
        if ltm_hits:                                                    # 若有命中
            self.stm.add("assistant", "[长期记忆节选]\n" + "\n".join(ltm_hits))  # 注入到上下文，帮助模型利用既有知识

        # 第2步：进入执行阶段（ReAct 风格：思考→工具→观测），直到不再调用工具为止
        self.stm.add("user", user_goal)                                 # 把本轮用户目标写入历史
        for _ in range(max_tool_rounds):                                 # 最多尝试若干次工具调用循环
            resp = self._call_model(self.stm.get(), with_tools=True)     # 允许函数调用
            msg = resp.choices[0].message                                # 取第一条回复
            tool_calls = getattr(msg, "tool_calls", None) or getattr(msg, "function_call", None)  # 兼容两种字段

            if not tool_calls:                                           # 若没有工具调用意图，则认为可以给出最终答案
                final_text = msg.content or ""                           # 直接取文本内容
                self.stm.add("assistant", final_text)                    # 写入历史
                return final_text                                        # 返回给调用者

            # 若存在工具调用，则规范化为列表逐一处理
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]

            # 执行每个工具调用，并将结果以 role=tool 写回历史，再次请求模型“综合回答”
            for tc in tool_calls:
                if hasattr(tc, "function"):                               # 新式结构：tc.function.name / tc.function.arguments
                    name = getattr(tc.function, "name", "")
                    args_json = getattr(tc.function, "arguments", "{}")
                    call_id = getattr(tc, "id", "")
                else:                                                     # 旧式结构：字典
                    name = tc.get("name", "")
                    args_json = tc.get("arguments", "{}")
                    call_id = tc.get("id", "")

                try:
                    args = json.loads(args_json) if isinstance(args_json, str) else (args_json or {})  # 解析参数
                except Exception:
                    args = {}                                             # 解析失败则给空参数

                impl = TOOL_IMPLS.get(name)                               # 根据名称找到本地实现
                if impl is None:                                          # 若未实现则返回错误信息
                    result = f"工具未实现：{name}"
                else:
                    try:
                        result = impl(**args)                             # 调用工具函数
                    except Exception as e:
                        result = f"工具执行异常：{type(e).__name__}: {e}"  # 失败时返回错误文本

                self._append_tool_result(call_id, name, str(result))      # 将工具结果写入历史以便模型“看见”

            # 工具结果已写回，继续让模型综合（本轮不再提供 tools，鼓励其产出自然语言）
            resp2 = self._call_model(self.stm.get(), with_tools=False)    # 不携带 tools，即不再触发新调用
            text2 = resp2.choices[0].message.content or ""                # 取综合后的文本回答
            if text2.strip():                                             # 若有回答文本
                self.stm.add("assistant", text2)                          # 写历史
                return text2                                              # 返回结果

            time.sleep(0.2)                                               # 轻微休眠，避免过快循环（演示节奏）

        # 若超过循环上限仍未产出，则给出降级提示
        fallback = "已达到工具调用上限，请精简问题或提供更多信息再试。"  # 构造回退文案
        self.stm.add("assistant", fallback)                               # 写入历史
        return fallback                                                   # 返回回退结果

# ==========================
#  命令行运行入口
# ==========================

def main():
    """简易 CLI：展示 Planner / Executor / Memory / Tools 的协同。"""
    agent = Agent(CLIENT, MODEL_NAME)                      # 实例化 Agent（聚合四大核心组件）
    print("Agent 已就绪：输入你的目标（/exit 退出）")     # 打印欢迎信息
    print("示例：读取 policy 文本、搜索 docs、计算表达式、询问当前时间等")  # 给出示例提示
    while True:                                            # 进入循环
        try:
            user = input("\n你：").strip()               # 读取用户输入
            if not user:                                   # 空输入则继续
                continue
            if user.lower() in ("/exit", "exit", "quit", "/quit"):  # 用户请求退出
                print("再见！")                         # 友好告别
                break                                      # 跳出循环
            answer = agent.run(user_goal=user)             # 调用 Agent 执行一次完整任务
            print(f"\nAgent：{answer}")                 # 打印返回内容
        except KeyboardInterrupt:                          # 捕获 Ctrl+C
            print("\n已中断，退出。")                      # 友好提示
            break                                          # 退出
        except Exception as e:                             # 其他异常
            print("发生错误：", e)                      # 打印错误
            traceback.print_exc()                          # 打印堆栈，便于调试

# ==========================
#  程序入口保护
# ==========================

if __name__ == "__main__":     # 仅当脚本直接执行时进入 main（被 import 时不执行）
    main()                     # 启动命令行交互
