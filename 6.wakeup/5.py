# -*- coding: utf-8 -*-
"""
持续监听 → faster-whisper/large-v3 → 检测唤醒词“你好，老丁”或“您好，老丁”
触发后用 SAPI5（优先 Huihui，其次 Zira）播报固定回复
播报完成后继续监听；若无继续询问，仍保持监听。

依赖：
  pip install faster-whisper==1.0.2 sounddevice==0.4.7 pyttsx3==2.90
"""

import time
import queue
import threading
import unicodedata
from typing import Optional, List

import numpy as np
import sounddevice as sd
import pyttsx3
from faster_whisper import WhisperModel

# ======== 可调参数 ========
SAMPLE_RATE = 16000
CHANNELS = 1
WINDOW_SECONDS = 2.5
HOP_SECONDS = 1.0

# 修改：支持多个唤醒词（统一做规范化后匹配）
WAKE_PHRASES: List[str] = ["你好老丁", "您好老丁"]

COOLDOWN_SECONDS = 4.0                   # 触发后冷却期
DEVICE_INDEX = None                      # 指定麦克风，None=默认
VOICE_PREFERRED = [
    "Microsoft Huihui Desktop - Chinese (Simplified)",
    "Microsoft Zira Desktop - English (United States)",
]
RESPONSE_TEXT = "你好，有什么可以帮您，回答来自可爱的老丁"

# 在 TTS 期间暂停识别，避免拾取自身播报声音
IS_SPEAKING = False
IS_SPEAKING_LOCK = threading.Lock()

audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)

def normalize_zh(text: str) -> str:
    """去空白与标点，转小写：例如 '你好，老丁！' -> '你好老丁'"""
    if not text:
        return ""
    buf = []
    for ch in text.lower():
        if ch.isspace():
            continue
        if unicodedata.category(ch).startswith("P"):  # 标点
            continue
        buf.append(ch)
    return "".join(buf)

def contains_wake_phrase(text: str, wakes: List[str] = WAKE_PHRASES) -> Optional[str]:
    """返回命中的唤醒词（规范化后子串匹配）；未命中返回 None。"""
    norm = normalize_zh(text)
    for w in wakes:
        if w in norm:
            return w
    return None

def pick_sapi5_voice(engine: pyttsx3.Engine, preferred_names=VOICE_PREFERRED) -> Optional[str]:
    voices = engine.getProperty("voices")
    # 精确或包含匹配
    for name in preferred_names:
        for v in voices:
            if v.name == name or (name.lower() in v.name.lower()):
                return v.id
    # 退而求其次：找中文/默认
    for v in voices:
        if ("Chinese" in v.name) or ("Huihui" in v.name):
            return v.id
    return voices[0].id if voices else None

def speak_sapi5(text: str):
    """同步播报：进入-离开时设置 IS_SPEAKING 标志，播报完成即恢复识别。"""
    global IS_SPEAKING
    with IS_SPEAKING_LOCK:
        IS_SPEAKING = True
    try:
        engine = pyttsx3.init(driverName="sapi5")
        vid = pick_sapi5_voice(engine)
        if vid:
            engine.setProperty("voice", vid)
        engine.setProperty("rate", 185)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    finally:
        time.sleep(0.3)  # 给尾音一点缓冲，避免被再次拾取
        with IS_SPEAKING_LOCK:
            IS_SPEAKING = False

def speak_async(text: str):
    """异步播报，不阻塞识别主循环。"""
    t = threading.Thread(target=speak_sapi5, args=(text,), daemon=True)
    t.start()

def audio_stream_worker():
    """采集线程：滚动窗口切片送入队列"""
    frame_len = int(HOP_SECONDS * SAMPLE_RATE)
    window_len = int(WINDOW_SECONDS * SAMPLE_RATE)
    ring = np.zeros(window_len, dtype=np.float32)
    try:
        with sd.InputStream(
            device=DEVICE_INDEX,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=frame_len,
        ) as stream:
            while True:
                frames, _ = stream.read(frame_len)
                mono = frames[:, 0].astype(np.float32, copy=False)
                ring = np.concatenate([ring[frame_len:], mono])
                try:
                    audio_q.put_nowait(ring.copy())
                except queue.Full:
                    _ = audio_q.get_nowait()
                    audio_q.put_nowait(ring.copy())
    except Exception as e:
        print(f"[Audio] 采集出错：{e}")

def build_model():
    device = "cpu"
    compute_type = "int8"
    try:
        import torch
        if torch.cuda.is_available():
            device, compute_type = "cuda", "float16"
    except Exception:
        pass
    print(f"[ASR] 加载 faster-whisper/large-v3, device={device}, compute_type={compute_type}")
    return WhisperModel("faster-whisper/large-v3", device=device, compute_type=compute_type)

def main():
    # 启动采集线程
    threading.Thread(target=audio_stream_worker, daemon=True).start()

    model = build_model()
    last_trigger_ts = 0.0
    print("[ASR] 持续监听中（Ctrl+C 退出）...")

    vad_params = dict(min_silence_duration_ms=300, speech_pad_ms=50)

    while True:
        try:
            audio = audio_q.get(timeout=1.0)

            # 如果当前在播报，跳过识别，保持持续监听
            with IS_SPEAKING_LOCK:
                if IS_SPEAKING:
                    continue

            segments, info = model.transcribe(
                audio,
                language="zh",
                task="transcribe",
                vad_filter=True,
                vad_parameters=vad_params,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                without_timestamps=True,
            )
            text = "".join(seg.text for seg in segments).strip()
            if not text:
                continue

            print(f"[ASR] 识别：{text}")

            hit = contains_wake_phrase(text)
            if hit:
                now = time.time()
                if now - last_trigger_ts >= COOLDOWN_SECONDS:
                    last_trigger_ts = now
                    # 还原成更友好的提示文案
                    printable = "你好，老丁" if hit == "你好老丁" else "您好，老丁"
                    print(f"[Wake] 检测到唤醒词：{printable}")
                    # 异步播报，播报完继续监听（无后续问题也持续监听）
                    speak_async(RESPONSE_TEXT)

            # 没触发就什么都不做 → 循环继续，始终监听
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            print("\n[Exit] 用户中断。")
            break
        except Exception as e:
            # 打印错误但不退出，保持“长时监听”特性
            print(f"[ASR] 异常：{e}")

if __name__ == "__main__":
    main()
