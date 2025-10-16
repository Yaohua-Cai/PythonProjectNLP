# -*- coding: utf-8 -*-
# ============================================================
#  RAG（检索增强生成）最小可用示例：本地 .txt 文档 + 智谱 GLM 对话生成
#  目标：演示完整 RAG 流程（收集→切分→索引→检索→组装上下文→生成带引用的回答）
#
#  依赖安装：
#    pip install zhipuai==2.* scikit-learn==1.* python-dotenv
#
#  运行前准备：
#    1) 设置环境变量：export ZHIPUAI_API_KEY="你的apikey"   # Windows: set ZHIPUAI_API_KEY=你的apikey
#    2) 在项目根目录创建 docs/ 目录，放置若干 .txt 文档（UTF-8 最佳）
#    3) 运行：python rag_local_txt_demo.py
#
#  说明：
#    - 本示例使用 TF-IDF（字符 n-gram）做轻量向量化，适配中英文，无需额外分词模型
#    - 仅作教学演示，生产实践建议改为：专用嵌入模型 + 更强检索（Hybrid/Rerank）+ 权限/新鲜度治理
# ============================================================

import os                                           # 操作系统相关（读取环境变量、路径拼接）
import glob                                         # 文件通配符（批量读取 docs 目录下的文件）
import json                                         # JSON 读写（调试与可视化）
from dataclasses import dataclass                   # dataclass 简化数据结构的定义
from typing import List, Tuple, Dict, Any           # 类型注解，提升可读性
from zhipuai import ZhipuAI                         # 智谱AI 官方 Python SDK
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF 向量化器
from sklearn.metrics.pairwise import cosine_similarity        # 余弦相似度计算

# ===========================
# 环境与全局参数
# ===========================

API_KEY ="699fa03dbe994df4852607411ef48a00.lhkhZH3ycEnMj0TA"

MODEL_NAME = "glm-4"                                # 选择使用的对话模型（可替换为 glm-4-air / glm-4-flash / glm-4-long 等）
DOCS_DIR = "docs"                                   # 本地知识库目录（存放 .txt 文件）
CHUNK_SIZE = 500                                    # 文本切块大小（字符数），平衡语义完整性与检索粒度
CHUNK_OVERLAP = 100                                 # 切块重叠（字符数），缓解边界语义断裂
TOP_K = 5                                           # 检索返回的候选块数量
MAX_CONTEXT_CHARS = 3000                            # 拼接入提示词的上下文最大字符数（防止上下文过长）
TEMPERATURE = 0.2                                   # 生成温度（低温更稳健）
TOP_P = 0.9                                         # nucleus 采样阈值（多样性控制）

# ===========================
# 数据结构定义
# ===========================

@dataclass
class Chunk:                                         # 定义切片数据结构，便于携带元信息
    doc_id: int                                     # 文档编号（从 0 开始）
    doc_path: str                                   # 文档路径（含文件名）
    start: int                                      # 在原文中的起始字符索引
    end: int                                        # 在原文中的结束字符索引（不含）
    text: str                                       # 切片文本内容

# ===========================
# 工具函数：读取与切分
# ===========================

def read_all_txt_docs(docs_dir: str) -> List[str]:
    """读取 docs/ 目录下所有 .txt 文件，返回每个文件的完整文本（列表顺序稳定）。"""
    # 使用 glob 匹配所有 .txt 文件（排序后保证索引稳定）
    paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    # 若目录中没有 .txt，给出友好提示
    if not paths:
        raise FileNotFoundError(f"未在 {docs_dir}/ 目录下找到 .txt 文档，请先放置知识库文本。")
    texts = []                                     # 准备承载每个文件的全文内容
    for p in paths:                                 # 遍历每个文件路径
        with open(p, "r", encoding="utf-8", errors="ignore") as f:  # 打开文件（忽略编码错误）
            texts.append(f.read())                  # 读取全文并压入列表
    return texts                                    # 返回文本列表（与 paths 对应）

def sliding_window_chunks(text: str, size: int, overlap: int) -> List[Tuple[int, int, str]]:
    """将单个文本按滑动窗口切分为若干块，返回 (start, end, chunk_text) 列表。"""
    # 初始化切片结果列表
    chunks = []
    # 从 0 开始，步长为 size - overlap，保证相邻块有重叠部分
    i = 0
    while i < len(text):                            # 只要还没到文本末尾就继续
        start = i                                   # 当前块的起始位置
        end = min(i + size, len(text))              # 当前块的结束位置（不超过文本长度）
        chunk_text = text[start:end]                # 取出对应片段
        chunks.append((start, end, chunk_text))     # 保存片段的位置信息与文本
        if end == len(text):                        # 若已到末尾则停止
            break
        i = end - overlap                           # 下一个块的起点向前 overlap 个字符，形成重叠
    return chunks                                   # 返回所有切片信息

def build_corpus_and_chunks(docs_dir: str) -> Tuple[List[str], List[str], List[Chunk]]:
    """读取所有文档并切分，返回：文件路径列表、语料列表（按 chunk）、Chunk 元信息列表。"""
    # 枚举并排序 docs 目录下的所有 .txt 文件
    doc_paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    # 若没有文件则报错
    if not doc_paths:
        raise FileNotFoundError(f"未在 {docs_dir}/ 目录下找到 .txt 文档，请先放置知识库文本。")
    # 读取全文列表，与 doc_paths 一一对应
    full_texts = []
    for p in doc_paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            full_texts.append(f.read())
    # 对每个文档进行滑动切片，组装“语料 + 元信息”
    corpus = []                                     # 保存所有切片文本（用于向量化）
    chunk_meta: List[Chunk] = []                    # 保存切片的元信息（文件/位置）
    for doc_id, (p, txt) in enumerate(zip(doc_paths, full_texts)):
        for (start, end, chunk_text) in sliding_window_chunks(txt, CHUNK_SIZE, CHUNK_OVERLAP):
            corpus.append(chunk_text)               # 把该片段文本加入语料
            chunk_meta.append(Chunk(                 # 同步记录片段的来源与位置信息
                doc_id=doc_id,
                doc_path=p,
                start=start,
                end=end,
                text=chunk_text
            ))
    # 返回三元组：文档路径列表、切片语料列表、切片元信息列表
    return doc_paths, corpus, chunk_meta

# ===========================
# 索引器：TF-IDF 向量化 + 余弦检索
# ===========================

class TfidfIndexer:
    """使用 TF-IDF（字符 n-gram）构建检索索引，适配中英文混排。"""
    def __init__(self):
        # analyzer='char_wb' 表示基于字符的 n-gram（忽略单字噪声），2~4 元提升鲁棒性
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.matrix = None                          # 稀疏向量矩阵（每行对应一个切片）
        self.corpus: List[str] = []                 # 语料（切片文本）
        self.meta: List[Chunk] = []                 # 切片元信息

    def fit(self, corpus: List[str], meta: List[Chunk]):
        """根据切片语料拟合 TF-IDF 并生成索引矩阵。"""
        self.corpus = corpus                        # 保存语料
        self.meta = meta                            # 保存元信息
        self.matrix = self.vectorizer.fit_transform(corpus)  # 拟合并向量化所有切片

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
        """对查询进行向量化，与所有切片计算相似度，返回 Top-K 切片及相似度。"""
        if self.matrix is None:                     # 若尚未构建索引则报错
            raise RuntimeError("索引尚未构建，请先调用 fit()。")
        q_vec = self.vectorizer.transform([query])  # 将查询转为 TF-IDF 向量
        sims = cosine_similarity(q_vec, self.matrix)[0]  # 计算与所有切片的余弦相似度（得到一维数组）
        top_idx = sims.argsort()[::-1][:top_k]      # 从大到小排序，取前 top_k 个索引
        results = []                                 # 准备装载（切片元信息，分数）
        for i in top_idx:
            results.append((self.meta[i], float(sims[i])))
        return results                               # 返回检索结果列表

# ===========================
# 上下文拼接与提示词模板
# ===========================

def format_context(chunks: List[Tuple[Chunk, float]], max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, List[Dict[str, Any]]]:
    """把若干切片按分数从高到低拼接为一个上下文字符串，并生成引用清单。"""
    # 初始化可拼接的上下文与引用元数据列表
    ctx_parts: List[str] = []
    citations: List[Dict[str, Any]] = []
    used = 0                                         # 已使用字符计数
    for rank, (ck, score) in enumerate(chunks, start=1):
        header = f"[{rank}] 文件: {os.path.basename(ck.doc_path)} | 位置: {ck.start}-{ck.end} | 相似度: {score:.3f}\n"
        body = ck.text.strip().replace("\n", " ")    # 去除换行，避免提示词结构混乱
        piece = header + body                        # 将头部元信息与正文片段合并
        if used + len(piece) > max_chars:            # 若超出阈值，停止拼接（避免上下文过长）
            break
        ctx_parts.append(piece)                      # 加入上下文列表
        used += len(piece)                           # 更新已用字符数
        citations.append({                           # 记录一个引用条目（用于最终附注）
            "rank": rank,
            "file": os.path.basename(ck.doc_path),
            "span": f"{ck.start}-{ck.end}",
            "score": round(score, 3)
        })
    # 用分隔线拼接所有片段为一个完整上下文字符串
    context = "\n\n---\n\n".join(ctx_parts)
    return context, citations                        # 返回上下文文本与引用条目列表

def build_prompt(user_query: str, context: str) -> List[Dict[str, str]]:
    """构造对话消息列表：系统提示 + 用户提问（内含上下文）。"""
    # 系统提示：明确要求“基于上下文作答、缺证据则拒答、务必附引用编号”
    system = {
        "role": "system",
        "content": (
            "你是一个严格基于给定材料回答问题的助手。"
            "规则：仅使用提供的【上下文】信息作答；若证据不足，请明确说明“在给定资料中未找到答案”。"
            "请在答案末尾给出引用编号，如 [1][3]，对应上下文中的段落标头。"
        )
    }
    # 用户消息：把“问题 + 上下文”合并传入，模型据此生成有据可依的回答
    user = {
        "role": "user",
        "content": (
            f"问题：{user_query}\n\n"
            f"【上下文（按相关性排序）】\n{context}\n\n"
            "请基于以上上下文回答，并在末尾附上引用编号。"
        )
    }
    return [system, user]                            # 返回消息列表，供 chat.completions.create 使用

# ===========================
# 与智谱 GLM 对接进行生成
# ===========================

def call_glm(messages: List[Dict[str, str]]) -> str:
    """调用智谱 GLM 聊天接口，返回模型回答文本。"""
    client = ZhipuAI(api_key=API_KEY)               # 用环境变量中的密钥构建客户端
    resp = client.chat.completions.create(          # 调用“聊天补全”接口
        model=MODEL_NAME,                           # 指定模型
        messages=messages,                          # 传入消息列表（系统+用户）
        temperature=TEMPERATURE,                    # 设置温度（越低越稳）
        top_p=TOP_P                                 # nucleus 采样阈值
    )
    return resp.choices[0].message.content or ""    # 取第一条候选的文本内容（容错为空串）

# ===========================
# RAG（检索→生成）主流程封装
# ===========================

class LocalTxtRAG:
    """本地 .txt 知识库的 RAG 实现（切分/索引/检索/生成）。"""
    def __init__(self, docs_dir: str = DOCS_DIR):
        self.docs_dir = docs_dir                    # 保存知识库目录路径
        self.doc_paths: List[str] = []              # 文档路径列表（用于显示与溯源）
        self.corpus: List[str] = []                 # 切片文本列表（用于向量化）
        self.meta: List[Chunk] = []                 # 切片元信息列表（用于引用与定位）
        self.indexer = TfidfIndexer()               # 初始化 TF-IDF 索引器

    def build(self):
        """构建（或重建）RAG 索引：读取→切分→向量化。"""
        self.doc_paths, self.corpus, self.meta = build_corpus_and_chunks(self.docs_dir)  # 读取并切分
        self.indexer.fit(self.corpus, self.meta)    # 基于切片语料拟合 TF-IDF 索引
        # 可选：打印索引规模，便于用户确认
        print(f"已索引文档数：{len(self.doc_paths)}，切片数：{len(self.corpus)}")

    def answer(self, query: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """执行一次 RAG：检索 Top-K 切片 → 组装上下文 → 调用 GLM → 返回答案与引用。"""
        hits = self.indexer.search(query, top_k=top_k)        # 第一步：检索最相关的切片
        context, citations = format_context(hits, MAX_CONTEXT_CHARS)  # 第二步：拼接上下文 + 收集引用
        messages = build_prompt(query, context)               # 第三步：构造提示词（系统+用户）
        answer_text = call_glm(messages)                      # 第四步：调用 GLM 生成回答
        # 返回包含答案、引用与用于调试的上下文，便于 UI 展示或日志审计
        return {
            "query": query,
            "answer": answer_text,
            "citations": citations,
            "context_preview": context[:400] + ("..." if len(context) > 400 else "")  # 仅预览前 400 字
        }

# ===========================
# 命令行交互（可选）
# ===========================

def main():
    """命令行交互入口：构建索引 → 循环问答（输入 /exit 退出）。"""
    # 实例化 RAG 对象
    rag = LocalTxtRAG(DOCS_DIR)
    # 构建索引（首次/文档更新后都需要）
    rag.build()
    # 打印简单的欢迎与提示
    print("\n本地 RAG 已就绪：输入你的问题，或输入 /exit 退出。")
    print("提示：问题越具体（含关键词），检索效果越好；答案末尾会附引用编号。")
    # 开始循环问答
    while True:
        try:
            q = input("\n你：").strip()                  # 读取用户输入
            if not q:                                      # 空输入则继续
                continue
            if q.lower() in ("/exit", "exit", "quit", "/quit"):  # 用户请求退出
                print("再见！")
                break
            result = rag.answer(q, top_k=TOP_K)            # 执行一次 RAG 流程
            print("\n答案：\n" + result["answer"])       # 打印模型回答
            # 打印引用清单，方便核对来源
            print("\n引用：")
            for c in result["citations"]:
                print(f"  - [{c['rank']}] {c['file']} ({c['span']}) | 相似度={c['score']}")
        except KeyboardInterrupt:                           # 捕获 Ctrl+C 友好退出
            print("\n已中断，退出。")
            break
        except Exception as e:                              # 其他异常（例如网络问题）
            print("发生错误：", e)

# ===========================
# 程序入口
# ===========================

if __name__ == "__main__":  # 仅当直接运行本脚本时，执行 main()
    main()                  # 启动命令行交互
