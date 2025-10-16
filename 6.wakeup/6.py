# -*- coding: utf-8 -*-  # 指定源码编码为 UTF-8
"""
修复版语音助手（Windows）：
- faster-whisper/large-v3 持续监听唤醒词：“你好，老丁 / 您好，老丁”
- 命中唤醒词后，TTS 必说：“我在”（持久 TTS 线程 + win32com 回退）
- 进入对话：持续监听 → 静音判停 → ASR → Qwen3-1.7B 生成（屏蔽 tool/thought 标签）→ TTS 播报
- 回答后等待下一问（可配置超时），超时未开口则退回待机（重新从唤醒开始）
- 待机阶段：能量门限 + 自适应底噪（EMA）抑制空噪误识别
- 对话阶段：噪声标定 + 动态阈值（start/stop）自动适配底噪
"""

# =========================
# 标准库与第三方依赖
# =========================
import time                                  # 用于计时与延迟
import queue                                 # 跨线程队列（音频/指令传递）
import threading                             # 创建持久线程（TTS、音频采集）
import unicodedata                           # 用于识别/过滤 Unicode 标点
from typing import Optional, List            # 类型注解辅助
import re                                    # 文本正则清洗（屏蔽工具标签等）

import numpy as np                           # 数值计算（音频 RMS 等）
import sounddevice as sd                     # 麦克风流采集
import pyttsx3                               # SAPI5 语音合成（主路径）
from faster_whisper import WhisperModel      # faster-whisper 推理模型

# =========================
# Qwen 模型配置（本地）
# =========================
QWEN_MODEL_NAME = r"Qwen3-1.7B"   # ← 改为你的本地模型路径或已缓存名称
QWEN_MAX_NEW_TOKENS = 512                       # 生成长度上限
BAD_WORDS_IDS: List[List[int]] = []             # 生成阶段禁止输出的 token 片段

# =========================
# 全局语音/识别参数
# =========================
SAMPLE_RATE = 16000                              # 16k 采样，ASR 推荐
CHANNELS = 1                                     # 单声道
WINDOW_SECONDS = 2.4                              # 待机滑窗长度（秒）
HOP_SECONDS = 0.8                                 # 滑步（秒），越小越灵敏

WAKE_PHRASES = ["你好老丁", "您好老丁"]              # 规范化后的唤醒词（去标点/空白）
COOLDOWN_SECONDS = 2.0                            # 唤醒冷却（秒），防连触发
DEVICE_INDEX = None                               # 指定麦克风设备索引（None=默认）

VOICE_PREFERRED = [                               # SAPI5 首选音色顺序
    "Microsoft Huihui Desktop - Chinese (Simplified)",
    "Microsoft Zira Desktop - English (United States)",
]
REPLY_WAKE = "我在"                                # 唤醒后必答
REPLY_DIDNT_HEAR = "没有听清楚，请重新输入。"        # 未识别或过短时的提示
EXIT_PHRASES = {"退出", "结束", "再见", "拜拜", "停止", "结束对话"}  # 口语退出词
WAIT_NEXT_QUESTION_TIMEOUT = 8.0                  # 回答后等待下一问超时（秒）

# =========================
# 对话端点/VAD 参数（动态）
# =========================
UTT_FRAME_MS = 200                                # 能量判定片长（毫秒）
UTT_SILENCE_TO_END = 0.9                          # 连续静音判停（秒）
UTT_MAX_SECONDS = 30.0                            # 单段上限（秒）
NOISE_CALIB_SECONDS = 0.8                         # 进入对话前底噪标定（秒）
START_FACTOR = 4.0                                # 开口阈值=噪声RMS*系数
STOP_FACTOR = 2.5                                 # 结束阈值=噪声RMS*系数

# =========================
# 待机底噪自适应（EMA）
# =========================
STANDBY_NOISE_RMS = 0.004                         # 待机初始底噪估计
STANDBY_NOISE_EMA_ALPHA = 0.05                    # EMA 系数（0~1），越大越敏感
STANDBY_ENERGY_GATE = 2.5                         # 待机能量门限倍数：rms > baseline*倍数 才做 ASR

# =========================
# 自声抑制（TTS 播报期间暂停识别）
# =========================
IS_SPEAKING = False                               # True 表示 TTS 正在播报
IS_SPEAKING_LOCK = threading.Lock()               # 锁保证跨线程安全

# =========================
# 跨线程队列
# =========================
ring_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)   # 待机滑窗队列
raw_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)   # 对话原始帧队列

# =========================
# 文本清洗：屏蔽“思考/工具调用”内容
# =========================
def sanitize_llm_output(text: str) -> str:
    """移除 <think>/<thought>/<tool_call>/<tool_response> 等及其内容，避免被 TTS 播报。"""
    if not text:                                                              # 若为空直接返回
        return text
    # —— 块级标签（含变体）整体删除 ——                                          #
    patterns_block = [
        r"<think>.*?</think>",                                                # 思考块
        r"<THINK>.*?</THINK>",                                                # 大写思考块
        r"<thought>.*?</thought>",                                            # thought 块
        r"<tool_call\b[^>]*>.*?</tool_call>",                                 # 工具调用块
        r"<tool_response\b[^>]*>.*?</tool_response>",                         # 工具响应块
        r"<\|im_start\|>\s*assistant_thought\s*.*?<\|im_end\|>",              # ChatML 思考
        r"<\|assistant_thought\|>.*?(?=<\|assistant\|>|$)",                   # 另一种 ChatML
    ]
    for pat in patterns_block:                                                # 逐一清理
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)

    # —— 行级前缀（“思考:”“Thought:” 等）删除整行 ——                          #
    patterns_line = [
        r"^\s*(思考|思路|推理|草稿|推断|脑暴|中间推理|thought|chain of thought)\s*[:：].*$",
    ]
    for pat in patterns_line:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # —— 残留尖括号标签（仅名称，无内容）剔除 ——                                   #
    leftovers = [
        r"</?think>", r"</?THINK>", r"</?thought>",
        r"</?tool_call[^>]*>", r"</?tool_response[^>]*>",
        r"<\|im_start\|>", r"<\|im_end\|>", r"<\|assistant\|>",
    ]
    for pat in leftovers:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # —— 压缩多余空白 ——                                                         #
    text = re.sub(r"\n{3,}", "\n\n", text)                                     # 压缩多空行
    return text.strip()                                                        # 去首尾空白

# =========================
# 文本规范化与唤醒检测
# =========================
def normalize_zh(text: str) -> str:
    """去空白与标点、转小写；例：'你好，老丁！' → '你好老丁'。"""
    if not text:
        return ""
    buf = []
    for ch in text.lower():                                                    # 逐字符遍历
        if ch.isspace():                                                       # 跳过空白
            continue
        if unicodedata.category(ch).startswith("P"):                           # 跳过标点
            continue
        buf.append(ch)                                                         # 保留汉字/字母/数字
    return "".join(buf)

def contains_wake(text: str) -> Optional[str]:
    """检测是否包含任一唤醒词，命中返回该词，否则 None。"""
    norm = normalize_zh(text)                                                  # 先规范化
    for w in WAKE_PHRASES:                                                     # 遍历唤醒词表
        if w in norm:                                                          # 子串匹配
            return w
    return None

# =========================
# 持久 TTS 线程（修复“有时不说话”）
# =========================
class TTSManager:
    """持久化的 TTS 引擎线程：队列消费文本；失败自动回退到 win32com。"""

    def __init__(self):
        self.q: "queue.Queue[tuple[str, threading.Event]]" = queue.Queue()     # 任务队列（文本, 完成事件）
        self.thread = threading.Thread(target=self._loop, daemon=True)         # 后台线程
        self._engine = None                                                    # pyttsx3 引擎实例（线程内创建）
        self._stop = False                                                     # 停止标志
        self.thread.start()                                                    # 启动线程

    def _loop(self):
        """线程主循环：初始化引擎并消费播报任务。"""
        try:
            self._engine = pyttsx3.init(driverName="sapi5")                    # 线程内创建引擎
            vid = self._pick_voice(self._engine)                               # 选择音色
            if vid:
                self._engine.setProperty("voice", vid)                         # 设置音色
            self._engine.setProperty("rate", 185)                               # 语速
            self._engine.setProperty("volume", 1.0)                             # 音量
            print(f"[TTS] 引擎就绪，音色={vid}")                                # 打印日志
        except Exception as e:
            print(f"[TTS] 初始化 pyttsx3 失败：{e}")                            # 初始化失败日志
            self._engine = None                                                # 置空以触发回退

        while not self._stop:                                                  # 持续消费
            try:
                text, done_evt = self.q.get(timeout=0.5)                       # 取一条任务
            except queue.Empty:
                continue                                                       # 无任务继续等

            # —— 播报期间加自声抑制，避免被自己触发 ——                              #
            with IS_SPEAKING_LOCK:
                global IS_SPEAKING
                IS_SPEAKING = True

            ok = False                                                         # 记录是否成功播报
            try:
                if self._engine is not None:                                   # 主路径：pyttsx3
                    print("[TTS] pyttsx3 播报中…")
                    self._engine.say(text)                                     # 送入队列
                    self._engine.runAndWait()                                  # 阻塞直到朗读完成
                    ok = True                                                  # 成功
                else:
                    print("[TTS] 无 pyttsx3 引擎，准备回退 win32com")
            except Exception as e:
                print(f"[TTS] pyttsx3 播报异常：{e}")                           # 失败日志
                ok = False                                                     # 标记失败

            if not ok:
                # —— 回退：win32com 直连 SAPI ——                                     #
                try:
                    import win32com.client                                     # 动态导入
                    speaker = win32com.client.Dispatch("SAPI.SpVoice")         # 创建 SAPI
                    print("[TTS] 回退到 win32com SAPI 播报…")
                    speaker.Speak(text)                                        # 同步播报
                    ok = True                                                  # 成功
                except Exception as e:
                    print(f"[TTS] win32com 回退失败：{e}")                      # 回退失败

            time.sleep(0.30)                                                   # 尾音缓冲 300ms
            with IS_SPEAKING_LOCK:
                IS_SPEAKING = False                                            # 解除自声抑制
            done_evt.set()                                                     # 标记本次播报完成

    def speak(self, text: str):
        """同步接口：入队并等待本条播报完成（确保“我在”一定说出来）。"""
        evt = threading.Event()                                                # 完成事件
        self.q.put((text, evt))                                                # 入队一个任务
        evt.wait()                                                             # 阻塞等待播报完成

    def stop(self):
        """优雅停止 TTS 线程。"""
        self._stop = True                                                      # 置停止标志

    @staticmethod
    def _pick_voice(engine: pyttsx3.Engine) -> Optional[str]:
        """按优先名单挑选音色；若找不到则返回任一音色。"""
        try:
            voices = engine.getProperty("voices")                              # 获取可用音色
            for name in VOICE_PREFERRED:                                       # 优先匹配名单
                for v in voices:
                    if v.name == name or (name.lower() in v.name.lower()):
                        return v.id                                            # 命中返回
            for v in voices:                                                   # 次选：任何中文相关
                if ("Chinese" in v.name) or ("Huihui" in v.name):
                    return v.id
            return voices[0].id if voices else None                            # 兜底：任意音色
        except Exception:
            return None

# 全局 TTS 管理器实例（进程内唯一）
TTS = TTSManager()

# =========================
# 音频采集线程（待机 + 对话共用）
# =========================
def audio_stream_worker():
    """开启唯一的麦克风输入流，同时喂给待机滑窗队列与对话原始帧队列。"""
    frame_len = int(HOP_SECONDS * SAMPLE_RATE)                                 # 每块样本点数
    window_len = int(WINDOW_SECONDS * SAMPLE_RATE)                             # 滑窗总样本数
    ring = np.zeros(window_len, dtype=np.float32)                              # 初始化滑窗

    try:
        with sd.InputStream(                                                   # 打开输入流
            device=DEVICE_INDEX,                                               # 麦克风设备
            samplerate=SAMPLE_RATE,                                            # 采样率
            channels=CHANNELS,                                                 # 单声道
            dtype="float32",                                                   # float32
            blocksize=frame_len,                                               # 每次读取的帧
        ) as stream:
            while True:                                                        # 持续读取
                frames, _ = stream.read(frame_len)                             # 取一块音频
                mono = frames[:, 0].astype(np.float32, copy=False)             # 转单通道
                # —— 待机滑窗滚动 ——                                              #
                ring = np.concatenate([ring[frame_len:], mono])                # 拼接形成滑窗
                try:
                    ring_q.put_nowait(ring.copy())                             # 投喂滑窗
                except queue.Full:
                    _ = ring_q.get_nowait()                                    # 丢旧帧
                    ring_q.put_nowait(ring.copy())                             # 放新帧
                # —— 对话原始帧 ——                                                #
                try:
                    raw_q.put_nowait(mono.copy())                              # 投喂原始帧
                except queue.Full:
                    _ = raw_q.get_nowait()
                    raw_q.put_nowait(mono.copy())
    except Exception as e:
        print(f"[Audio] 采集出错：{e}")                                        # 设备异常打印

# =========================
# ASR 模型加载与转写
# =========================
def build_asr_model() -> WhisperModel:
    """加载 faster-whisper/large-v3，自动选择设备与精度。"""
    device = "cpu"                                                             # 缺省 CPU
    compute_type = "int8"                                                      # CPU 上 int8 更快
    try:
        import torch
        if torch.cuda.is_available():                                          # 若存在 CUDA
            device, compute_type = "cuda", "float16"                           # 用 float16
    except Exception:
        pass
    print(f"[ASR] faster-whisper/large-v3 → device={device}, compute_type={compute_type}")  # 打印设备
    return WhisperModel("faster-whisper/large-v3", device=device, compute_type=compute_type)  # 返回模型实例

def transcribe_ndarray(asr: WhisperModel, audio_f32: np.ndarray) -> str:
    """对 numpy float32 波形执行中文优先转写，返回识别文本。"""
    segments, _ = asr.transcribe(                                              # 调用转写
        audio_f32,
        language="zh",                                                         # 使用中文
        task="transcribe",                                                     # 纯转写
        vad_filter=True,                                                       # 开启内置 VAD
        vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=50),   # VAD 参数
        beam_size=1,                                                           # 速度优先
        best_of=1,                                                             # 单次最优
        temperature=0.0,                                                       # 贪心
        without_timestamps=True,                                               # 不返回时间戳
    )
    return "".join(seg.text for seg in segments).strip()                       # 合并文本

# =========================
# Qwen 加载与对话（带 tool/thought 屏蔽）
# =========================
def build_qwen():
    """加载本地 Qwen3-1.7B；构造 bad_words_ids 阻断工具/思考标签。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # 按设备设精度
    model = AutoModelForCausalLM.from_pretrained(                              # 加载模型
        QWEN_MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(                                 # 加载分词器
        QWEN_MODEL_NAME, trust_remote_code=True, local_files_only=True
    )

    # —— 构造 bad_words_ids：禁止生成这些片段（防思考/工具标签） ——                 #
    global BAD_WORDS_IDS
    forbidden_tokens = [
        "<think>", "</think>", "<THINK>", "</THINK>",
        "<thought>", "</thought>",
        "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
        "<tool_call", "<tool_response",  # 包含带属性标签起始
        "<|assistant_thought|>", "<|im_start|>assistant_thought", "<|im_end|>",
        "assistant_thought", "assistant_thought:",
    ]
    BAD_WORDS_IDS = []
    for tok in forbidden_tokens:
        ids = tokenizer(tok, add_special_tokens=False).input_ids               # 编码为 token 序列
        if ids:                                                                # 能编码的才加入
            BAD_WORDS_IDS.append(ids)

    return tokenizer, model

def qwen_chat(tokenizer, model, user_text: str) -> str:
    """Qwen Instruct 对话：禁止工具/思考标签 + 文本清洗，仅返回最终答案。"""
    import torch
    # —— 聊天提示，明确要求“不要输出任何标签/工具调用” ——                            #
    messages = [
        {"role": "system", "content": "你是中文助手。请直接输出最终答案，不要输出任何思考、工具调用或标签（如 <tool_call>、<tool_response> 等）。"},
        {"role": "user", "content": user_text},
    ]
    # —— 应用聊天模板，得到模型输入 ——                                              #
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    # —— 结束 token 配置 ——                                                       #
    eos_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "chat_template") and "<|im_end|>" in tokenizer.chat_template:
        if tokenizer.convert_tokens_to_ids is not None:
            stop_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if stop_id is not None:
                eos_ids.append(stop_id)

    # —— 推理生成（屏蔽 bad_words_ids） ——                                          #
    with torch.no_grad():
        out = model.generate(
            prompt,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=eos_ids,
            bad_words_ids=BAD_WORDS_IDS if BAD_WORDS_IDS else None,  # 关键：禁止输出标签
        )
    # —— 解码新生成内容，并做二次清洗 ——                                           #
    gen = out[0][prompt.shape[-1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    text = sanitize_llm_output(text)                                           # 正则清洗
    return text.strip()

# =========================
# 工具函数：清队列 / 噪声标定 / 动态端点检测
# =========================
def drain_queue(q: queue.Queue):
    """尽量清空队列，避免陈旧数据影响判定。"""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass

def calibrate_noise_baseline(duration_s: float = NOISE_CALIB_SECONDS) -> float:
    """在对话开始前采样底噪，返回 RMS 中位数（抗异常），并打印日志。"""
    deadline = time.time() + duration_s
    rms_vals = []
    frame_len = int((UTT_FRAME_MS / 1000.0) * SAMPLE_RATE)
    while time.time() < deadline:
        try:
            frame = raw_q.get(timeout=0.2)
        except queue.Empty:
            continue
        with IS_SPEAKING_LOCK:
            if IS_SPEAKING:                                                    # 播报期的数据不计入
                continue
        idx = 0
        while idx < len(frame):
            chunk = frame[idx: idx + frame_len]
            idx += frame_len
            if len(chunk) == 0:
                break
            rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
            rms_vals.append(rms)
    if not rms_vals:
        return 0.005                                                           # 安全默认
    baseline = float(np.median(rms_vals))                                      # 中位数抗干扰
    print(f"[VAD] 噪声RMS基线={baseline:.5f}")
    return max(baseline, 1e-4)                                                 # 不低于 1e-4

def wait_and_capture_one_utterance_dynamic(
    start_rms: float,
    stop_rms: float,
    start_timeout: Optional[float] = None
) -> Optional[np.ndarray]:
    """
    动态阈值端点检测：
    - 未开口：无限等/或超时等到能量 > start_rms 开始收集
    - 已开口：收集直到静音累计 >= UTT_SILENCE_TO_END 或时长 >= UTT_MAX_SECONDS
    """
    frame_len = int((UTT_FRAME_MS / 1000.0) * SAMPLE_RATE)
    collected = []
    started = False
    silence_acc = 0.0
    t0 = time.time()
    while True:
        if (not started) and (start_timeout is not None) and (time.time() - t0 > start_timeout):
            return None                                                         # 超时未开口
        try:
            frame = raw_q.get(timeout=0.5)
        except queue.Empty:
            continue
        with IS_SPEAKING_LOCK:
            if IS_SPEAKING:                                                     # 播报中跳过数据
                continue
        idx = 0
        while idx < len(frame):
            chunk = frame[idx: idx + frame_len]
            idx += frame_len
            if len(chunk) == 0:
                break
            rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
            if not started:
                if rms >= start_rms:                                            # 触发开口
                    started = True
                    collected.append(chunk)
                    silence_acc = 0.0
            else:
                collected.append(chunk)
                if rms < stop_rms:                                             # 静音累积
                    silence_acc += (len(chunk) / SAMPLE_RATE)
                else:
                    silence_acc = 0.0                                          # 有语音则清零
                dur = len(np.concatenate(collected)) / SAMPLE_RATE             # 当前段落时长
                if silence_acc >= UTT_SILENCE_TO_END or dur >= UTT_MAX_SECONDS:
                    return np.concatenate(collected).astype(np.float32)        # 返回整段

# =========================
# 对话模式：首问无限等待；回答后限时等待
# =========================
def dialog_mode_loop(asr: WhisperModel, tokenizer, qwen):
    """进入对话：噪声标定→动态阈值；回答后限时等待下一问，超时退回待机。"""
    print("[Dialog] 进入对话模式：持续监听，你说完我就回答。")
    drain_queue(raw_q)                                                         # 清除陈旧帧
    noise_rms = calibrate_noise_baseline(NOISE_CALIB_SECONDS)                  # 标定底噪
    start_rms = noise_rms * START_FACTOR                                       # 开口阈值
    stop_rms = noise_rms * STOP_FACTOR                                         # 结束阈值
    print(f"[VAD] 阈值：start={start_rms:.5f}, stop={stop_rms:.5f}")

    # —— 第一问：无限等待开口 ——                                                     #
    first_audio = wait_and_capture_one_utterance_dynamic(
        start_rms=start_rms, stop_rms=stop_rms, start_timeout=None
    )
    if first_audio is None:                                                    # 理论不至于发生
        return

    first_text = transcribe_ndarray(asr, first_audio)                          # 识别文本
    if not first_text or len(normalize_zh(first_text)) <= 1:                   # 无效文本
        print("[ASR] 未能识别出有效文本（首问）。")
        TTS.speak(REPLY_DIDNT_HEAR)                                            # 语音提示
        return                                                                 # 回待机

    print(f"[User] {first_text}")
    norm = normalize_zh(first_text)
    if any(p in norm for p in EXIT_PHRASES):                                   # 首句即退出
        TTS.speak("好的，已退出。")
        print("[Dialog] 收到退出指令，返回待机。")
        return

    try:
        answer = qwen_chat(tokenizer, qwen, first_text)                        # Qwen 生成
    except Exception as e:
        print(f"[LLM] 生成失败：{e}")
        answer = "抱歉，生成回答时出现问题。"
    print(f"[Qwen] {answer}")
    TTS.speak(answer)                                                          # 播报答案

    # —— 连续轮：回答后限时等待下一问 ——                                              #
    while True:
        next_audio = wait_and_capture_one_utterance_dynamic(
            start_rms=start_rms,
            stop_rms=stop_rms,
            start_timeout=WAIT_NEXT_QUESTION_TIMEOUT                           # 限时等待
        )
        if next_audio is None:                                                 # 超时未开口
            print("[Dialog] 等待下一问超时，返回待机。")
            return                                                             # 退回待机

        next_text = transcribe_ndarray(asr, next_audio)                        # 识别文本
        if not next_text or len(normalize_zh(next_text)) <= 1:                 # 无效文本
            print("[ASR] 未能识别出有效文本（后续）。")
            TTS.speak(REPLY_DIDNT_HEAR)                                        # 语音提示
            continue                                                           # 继续等待下一问

        print(f"[User] {next_text}")
        norm2 = normalize_zh(next_text)
        if any(p in norm2 for p in EXIT_PHRASES):                              # 退出词
            TTS.speak("好的，已退出。")
            print("[Dialog] 收到退出指令，返回待机。")
            return

        try:
            answer2 = qwen_chat(tokenizer, qwen, next_text)                    # 继续生成
        except Exception as e:
            print(f"[LLM] 生成失败：{e}")
            answer2 = "抱歉，生成回答时出现问题。"
        print(f"[Qwen] {answer2}")
        TTS.speak(answer2)                                                     # 播报答案

# =========================
# 主循环：待机监听→唤醒→说“我在”→对话→回待机
# =========================
def main():
    """程序入口：拉起音频与模型，执行待机→对话→待机循环。"""
    threading.Thread(target=audio_stream_worker, daemon=True).start()          # 开启采集线程

    asr = build_asr_model()                                                    # 加载 ASR
    try:
        tokenizer, qwen = build_qwen()                                         # 加载 Qwen
        print("[LLM] Qwen3-1.7B 已加载。")
    except Exception as e:
        print(f"[LLM] 加载失败：{e}\n请确认 QWEN_MODEL_NAME 指向本地可用模型。")
        tokenizer = qwen = None

    print("[ASR] 正在持续监听（Ctrl+C 退出）...")
    last_trigger_ts = 0.0                                                      # 冷却计时

    global STANDBY_NOISE_RMS                                                   # 使用/更新待机底噪

    while True:
        try:
            ring = ring_q.get(timeout=1.0)                                     # 取一帧滑窗
            with IS_SPEAKING_LOCK:
                if IS_SPEAKING:                                                # 播报期间不识别
                    continue

            # —— 动态更新待机底噪的 EMA ——                                            #
            ring_rms = float(np.sqrt(np.mean(ring**2) + 1e-12))                # 计算滑窗 RMS
            STANDBY_NOISE_RMS = (1.0 - STANDBY_NOISE_EMA_ALPHA) * STANDBY_NOISE_RMS \
                                + STANDBY_NOISE_EMA_ALPHA * ring_rms           # EMA 更新

            # —— 能量门限：能量过低（≈底噪）则跳过 ASR，减少噪声误识别 ——                #
            energy_gate = max(STANDBY_NOISE_RMS * STANDBY_ENERGY_GATE, 0.003)  # 动态门限，含下限
            if ring_rms < energy_gate:                                         # 能量不足 → 多半是底噪
                continue                                                       # 不做 ASR，节能稳健

            # —— 做一次轻量转写用于唤醒检测 ——                                         #
            segments, _ = asr.transcribe(
                ring,
                language="zh",
                task="transcribe",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=50),
                beam_size=1,
                best_of=1,
                temperature=0.0,
                without_timestamps=True,
            )
            text = "".join(seg.text for seg in segments).strip()               # 合并文本
            if not text:
                continue

            print(f"[ASR] 识别：{text}")                                       # 打印识别日志
            hit = contains_wake(text)                                          # 检测唤醒词
            if not hit:
                continue

            now = time.time()
            if now - last_trigger_ts < COOLDOWN_SECONDS:                       # 冷却中忽略
                continue
            last_trigger_ts = now                                              # 更新冷却计时

            printable = "你好，老丁" if hit == "你好老丁" else "您好，老丁"              # 友好输出
            print(f"[Wake] 检测到唤醒词：{printable}")                           # 唤醒日志

            # —— 必须播报“我在”（持久 TTS 线程确保可靠发声） ——                          #
            TTS.speak(REPLY_WAKE)                                              # 同步等待播报完成

            # —— LLM 准备检查 ——                                                      #
            if tokenizer is None or qwen is None:
                TTS.speak("本地大语言模型暂未就绪，请检查配置。")
                continue

            # —— 进入对话模式（动态阈值、静音判停、回答后限时等待下一问） ——                #
            dialog_mode_loop(asr, tokenizer, qwen)                              # 对话结束后返回
            print("[State] 回到待机，继续等待唤醒。")                               # 状态提示

        except queue.Empty:
            continue
        except KeyboardInterrupt:
            print("\n[Exit] 用户中断。")
            break
        except Exception as e:
            print(f"[Loop] 异常：{e}")                                          # 打印异常但保持运行

# =========================
# 程序入口
# =========================
if __name__ == "__main__":
    main()  # 启动主循环
