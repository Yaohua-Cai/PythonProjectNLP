# -*- coding: utf-8 -*-
"""
功能：将 YOLOv8 模型导出为 ONNX 并保存（含可选简化与一次性校验）
环境建议：
  pip install ultralytics onnx onnxruntime onnxsim
说明：
  - 默认导出 yolov8n（最小模型），可改为你自己的 .pt 权重路径
  - 支持动态 batch/尺寸、opset 选择、是否半精度导出
  - 导出完成后用 onnxruntime 做一次前向校验（随机输入），确认模型能跑通
"""

import os
import sys
import platform
from pathlib import Path

# -------- 可配置参数 --------
MODEL_SOURCE = "yolov8n.pt"       # 可填：'yolov8n.pt' / 你的本地权重 'runs/train/exp/weights/best.pt'
IMG_SIZE = (640, 640)             # 导出输入尺寸 (h, w)
BATCH = 1                         # 导出时的 batch 维度（若开启 dynamic，不强制固定）
DYNAMIC = True                    # 是否启用动态维度（动态 batch/height/width）
SIMPLIFY = True                   # 是否使用 onnx-simplifier 简化图
OPSET = 12                        # ONNX opset，通常 12/13/17 均可；TensorRT 常用 12 或 13
HALF = False                      # 是否半精度导出（部分后处理算子不支持 half，通常保持 False）
DEVICE = 0                        # 使用 GPU 序号；无 GPU 或想用 CPU 则设为 'cpu'

# -------- 1) 导出为 ONNX --------
def export_yolov8_to_onnx():
    # 延迟导入 ultralytics，避免无关环境报错
    from ultralytics import YOLO

    # 1.1 加载 YOLOv8 模型（可自动下载官方预训练权重）
    model = YOLO(MODEL_SOURCE)  # 如果传入自训 best.pt，会直接加载

    # 1.2 导出
    # ultralytics 的导出会在 runs/weights/ 目录下生成 onnx 文件，返回导出文件路径
    onnx_path = model.export(
        format="onnx",            # 导出格式
        imgsz=IMG_SIZE,           # 输入尺寸 (h, w) 或 int
        batch=BATCH,              # batch 维度
        dynamic=DYNAMIC,          # 动态维度（动态 batch/尺寸）
        simplify=SIMPLIFY,        # 是否简化（若本地安装 onnxsim 则会生效）
        opset=OPSET,              # ONNX opset 版本
        half=HALF,                # 半精度（多数部署链路不建议打开）
        device=DEVICE             # 导出时用的设备
        # 可选项：optimize=True（开 XNNPACK 优化，CPU 友好），int8=True（需量化流程）
    )

    print(f"[OK] 导出完成：{onnx_path}")
    return Path(onnx_path)

# -------- 2) 用 onnxruntime 进行一次快速校验（随机输入）--------
def quick_ort_check(onnx_file: Path):
    import onnx
    import numpy as np
    import onnxruntime as ort

    # 2.1 结构合法性检查
    onnx_model = onnx.load(onnx_file.as_posix())
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX 结构检查通过")

    # 2.2 创建推理会话（尽量用 GPU；若不可用自动回退 CPU）
    providers = []
    try:
        # CUDAExecutionProvider 可用时置前
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(onnx_file.as_posix(), providers=providers)
    except Exception:
        providers = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(onnx_file.as_posix(), providers=providers)

    print(f"[INFO] onnxruntime providers: {sess.get_providers()}")

    # 2.3 构造随机输入（形状需与导出一致，动态模型可任意正整尺寸）
    input_name = sess.get_inputs()[0].name
    # YOLOv8 ONNX 通常输入为 NCHW、float32、[0,1] 归一化；这里用随机数模拟
    H, W = IMG_SIZE
    dummy = (np.random.rand(BATCH, 3, H, W).astype("float32"))

    # 2.4 前向一次，打印输出信息（不同导出配置输出节点可能数量不同）
    outputs = sess.run(None, {input_name: dummy})
    print(f"[OK] 前向推理成功，输出张量个数: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  - outputs[{i}] shape = {tuple(out.shape)}, dtype={out.dtype}")

    # 小提示：Ultralytics 默认导出的 YOLOv8-ONNX 通常已包含后处理（NMS），
    # 也可能导出 raw logits（取决于版本/参数）。具体依赖你导出时的 ultralytics 版本。
    # 若得到的是 NMS 后结果，输出通常为 [num, 6] 或 [batch, num, 6]（xyxy, score, cls）。
    # 若是 raw 输出，形状类似 [batch, anchors, classes+box_dims]。

# -------- 主入口 --------
if __name__ == "__main__":
    print(f"[INFO] Python {sys.version.split()[0]}, OS: {platform.system()} {platform.release()}")
    print(f"[INFO] 配置：MODEL_SOURCE={MODEL_SOURCE}, IMG_SIZE={IMG_SIZE}, BATCH={BATCH}, "
          f"DYNAMIC={DYNAMIC}, SIMPLIFY={SIMPLIFY}, OPSET={OPSET}, HALF={HALF}, DEVICE={DEVICE}")

    onnx_file = export_yolov8_to_onnx()

    # 可选：做一次快速校验，确认 ONNX 能正常前向
    try:
        quick_ort_check(onnx_file)
    except Exception as e:
        print("[WARN] onnxruntime 校验失败（不影响导出文件存在）：", repr(e))

    print(f"[DONE] 已生成：{onnx_file.resolve()}")
    print("      你可以用 onnxruntime / OpenVINO / TensorRT 等继续部署。")
