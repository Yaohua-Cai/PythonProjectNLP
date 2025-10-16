# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8，防止中文注释/字符串乱码
# YOLOv8(ONNX) + 摄像头 + 多边形ROI + "person"进入计数 + CSV + 中文正确显示（Pillow）  # 本脚本功能说明

import time  # 用于计算耗时和估算 FPS
import csv  # 用于将事件数据写入 CSV 文件
from datetime import datetime  # 用于生成时间戳字符串
from pathlib import Path  # 跨平台地处理文件路径
import cv2  # OpenCV，用于视频采集、图像处理与绘制
import numpy as np  # 数值计算库，处理矩阵/向量
import onnxruntime as ort  # ONNXRuntime，执行 ONNX 模型推理
from PIL import Image, ImageDraw, ImageFont  # Pillow，用于在图像上正确渲染中文

# ============================ 配置区域 ============================
ONNX_PATH = "yolov8n.onnx"              # YOLOv8 的 ONNX 模型路径（请替换为你的模型）
IMG_SIZE = (640, 640)                   # 推理输入分辨率 (width, height)，需与导出时一致
CONF_THRES = 0.25                       # 原始输出解码时的置信度阈值
IOU_THRES = 0.45                        # 原始输出解码时的 NMS IoU 阈值
USE_CUDA = True                         # 是否优先使用 GPU（需安装 onnxruntime-gpu）
CSV_PATH = "person_counts.csv"          # 事件日志 CSV 文件路径
MIN_BOX_AREA = 15 * 15                  # 过滤极小目标的像素面积阈值
FONT_PATH = None                        # 中文字体路径；None 表示自动探测常见系统字体
FONT_SIZE_SMALL = 22                    # 小号字体大小（像素）
FONT_SIZE_MED = 24                      # 中号字体大小（像素）
FONT_SIZE_BIG = 26                      # 大号字体大小（像素）

# COCO 类别名（只统计 person=0）  # YOLOv8 默认训练在 COCO80 类
COCO_CLASSES = [  # 列表中索引即类别 ID
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]  # 以上为 COCO80 类名，person 的类别 ID 为 0

# ============================ 中文文本渲染（Pillow） ============================
def _auto_pick_font():  # 自动选择常见中文字体文件
    """自动选择常见中文字体；找不到则返回 None。"""  # 函数文档字符串
    candidates = [  # 常见系统字体路径候选（按平台）
        "C:/Windows/Fonts/simhei.ttf",                 # Windows: 黑体
        "C:/Windows/Fonts/msyh.ttc",                   # Windows: 微软雅黑
        "/System/Library/Fonts/PingFang.ttc",          # macOS: 苹方
        "/System/Library/Fonts/STHeiti Medium.ttc",    # macOS: 华文黑体
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux: Noto CJK
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux: Noto CJK
    ]  # 字体候选列表结束
    for p in candidates:  # 遍历候选路径
        if Path(p).exists():  # 如果该字体文件存在
            return p  # 返回可用的字体路径
    return None  # 若均不存在则返回 None

def put_text_cn(img_bgr, text, org, font_path=None, font_size=22, color=(0,255,255), bold=True):  # 在 OpenCV 图像上绘制中文
    """
    在 OpenCV 图像上正确渲染中文文本（支持描边“加粗”）。  # 功能说明
    img_bgr : OpenCV BGR 图像（原地修改）  # 参数说明
    text    : 文本内容  # 参数说明
    org     : 左上角坐标 (x, y)  # 参数说明
    font_path: 字体路径；None 时自动探测  # 参数说明
    font_size: 字号（像素）  # 参数说明
    color   : 文本颜色（BGR）  # 参数说明
    bold    : 是否绘制黑色描边增强可读性  # 参数说明
    """  # 文档字符串结束
    if font_path is None:  # 若未显式传入字体路径
        font_path = _auto_pick_font()  # 自动探测常见中文字体
    if font_path is None or not Path(font_path).exists():  # 若仍未找到字体文件
        raise FileNotFoundError("未找到中文字体，请设置 FONT_PATH 为本机字体文件路径。")  # 抛出异常提示配置字体

    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))  # 将 OpenCV BGR 转为 PIL RGB 图像
    draw = ImageDraw.Draw(img_pil)  # 获取 PIL 的绘图句柄
    font = ImageFont.truetype(font_path, font_size)  # 加载指定大小的 TrueType 字体

    x, y = org  # 解包文本起始坐标
    if bold:  # 若需要“加粗效果”
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:  # 八邻域描边偏移
            draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0))  # 绘制黑色描边
    draw.text((x, y), text, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))  # 绘制正文（RGB 顺序）

    img_bgr[:, :, :] = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)  # 将 PIL 图像写回 OpenCV BGR（原地修改）
    return img_bgr  # 返回修改后的图像引用（可链式调用）

# ============================ YOLOv8 ONNX 推理与后处理 ============================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):  # YOLO 常用的等比缩放+填边
    shape = im.shape[:2]  # 原图高宽 (h, w)
    if isinstance(new_shape, int):  # 若传入单整数
        new_shape = (new_shape, new_shape)  # 转为方形尺寸 (w, h)
    new_w, new_h = new_shape  # 目标宽高
    r = min(new_w / shape[1], new_h / shape[0])  # 计算等比缩放比例
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # 缩放后未填充的尺寸 (w, h)
    dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]  # 需要填充的宽高差
    dw /= 2; dh /= 2  # 左右与上下对半填充
    if shape[::-1] != new_unpad:  # 若需要缩放
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)  # 双线性插值缩放
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下填充像素
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右填充像素
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 用常数色填充
    return im, r, (left, top)  # 返回处理后的图、缩放比例 r、左上角 pad 偏移

def nms_boxes(boxes, scores, iou_thres=0.5):  # 纯 NumPy 的 NMS 实现
    x1, y1, x2, y2 = boxes.T  # 拆分为四个坐标向量
    areas = (x2 - x1) * (y2 - y1)  # 计算每个框面积
    order = scores.argsort()[::-1]  # 得分从高到低的索引
    keep = []  # 保留的索引列表
    while order.size > 0:  # 迭代直到没有候选
        i = order[0]  # 取当前最高分的索引
        keep.append(i)  # 加入保留列表
        if order.size == 1: break  # 若只剩一个候选，直接结束
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 交集左上 x
        yy1 = np.maximum(y1[i], y1[order[1:]])  # 交集左上 y
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 交集右下 x
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 交集右下 y
        w = np.maximum(0.0, xx2 - xx1)  # 交集宽（小于0则置0）
        h = np.maximum(0.0, yy2 - yy1)  # 交集高（小于0则置0）
        inter = w * h  # 交集面积
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)  # IoU 计算（加微量防除零）
        inds = np.where(iou <= iou_thres)[0]  # 保留 IoU 小于阈值的索引位置
        order = order[inds + 1]  # 更新剩余候选的索引序列
    return keep  # 返回保留索引

def scale_coords(boxes, ratio, pad):  # 将 letterbox 坐标映射回原图
    boxes[:, [0, 2]] -= pad[0]  # 去除 x 方向的左侧填充
    boxes[:, [1, 3]] -= pad[1]  # 去除 y 方向的上侧填充
    boxes[:, :4] /= ratio  # 再除以缩放比例 r
    return boxes  # 返回映射后的坐标

def xywh2xyxy(xywh):  # (cx,cy,w,h) -> (x1,y1,x2,y2)
    xyxy = np.zeros_like(xywh)  # 预分配输出数组
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1 = cx - w/2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1 = cy - h/2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2 = cx + w/2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2 = cy + h/2
    return xyxy  # 返回转换后的坐标

class YOLOv8ONNX:  # 封装 YOLOv8 ONNX 推理，统一输出格式
    """统一输出 [N,6]: x1,y1,x2,y2,score,cls（原图尺度）。"""  # 类文档
    def __init__(self, onnx_path, use_cuda=True):  # 构造函数
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]  # 选择推理后端
        try:  # 尝试按配置创建会话
            self.session = ort.InferenceSession(Path(onnx_path).as_posix(), providers=providers)  # 创建 ORT 会话
        except Exception:  # 若失败（如无 GPU 版 ORT）
            self.session = ort.InferenceSession(Path(onnx_path).as_posix(), providers=["CPUExecutionProvider"])  # 回退到 CPU
        self.input_name = self.session.get_inputs()[0].name  # 记录第一个输入张量的名称
        self.outputs_info = [(o.name, o.shape) for o in self.session.get_outputs()]  # 记录输出节点信息（调试用）

    def infer(self, img_bgr):  # 对单帧 BGR 图像做推理
        img0 = img_bgr  # 保存原始引用
        lb_img, ratio, pad = letterbox(img0, (IMG_SIZE[0], IMG_SIZE[1]))  # 预处理：等比缩放+填边
        img_rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
        img = img_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # HWC->CHW，扩 batch 维，归一化
        outputs = self.session.run(None, {self.input_name: img})  # 执行前向推理，取所有输出

        dets = None  # 初始化检测结果容器
        if len(outputs) == 1:  # 如果只有一个输出节点
            out = outputs[0]  # 取该输出
            if out.ndim == 2 and out.shape[1] in (6, 7):  # 已含 NMS 的二维输出
                if out.shape[1] == 7: out = out[:, 1:]  # 去掉 batch_id 列
                dets = out  # 直接作为结果
            elif out.ndim == 3 and out.shape[-1] in (6, 7):  # [1,N,6/7] 形态
                out = out[0]  # 去掉 batch 维
                if out.shape[1] == 7: out = out[:, 1:]  # 去掉 batch_id
                dets = out  # 直接作为结果
            elif out.ndim == 3 and out.shape[0] == 1 and out.shape[1] >= 5:  # 原始输出 [1,C,Num]
                dets = self._postprocess_raw(out[0], ratio, pad)  # 本地解码+NMS
        else:  # 多输出节点的情况
            det = None  # 临时候选
            for o in outputs:  # 优先搜寻已含 NMS 的输出
                if o.ndim == 2 and o.shape[1] in (6, 7):  # 二维已含 NMS
                    det = o if o.shape[1] == 6 else o[:, 1:]  # 去 batch_id
                    break  # 找到就停止
                if o.ndim == 3 and o.shape[-1] in (6, 7):  # 三维 [1,N,6/7]
                    det = o[0] if o.shape[-1] == 6 else o[0][:, 1:]  # 去 batch 维/列
                    break  # 找到就停止
            if det is not None:  # 若找到了已含 NMS 的输出
                dets = det  # 直接使用
            else:  # 否则尝试按原始输出解码
                for o in outputs:  # 再遍历寻找 [1,C,Num]
                    if o.ndim == 3 and o.shape[0] == 1 and o.shape[1] >= 5:  # 满足原始输出特征
                        dets = self._postprocess_raw(o[0], ratio, pad)  # 本地后处理
                        break  # 停止搜索

        if dets is None or dets.size == 0:  # 若没有检测结果
            return np.zeros((0, 6), dtype=np.float32)  # 返回空数组
        boxes = dets[:, :4].copy()  # 取出框坐标
        scores = dets[:, 4]  # 取出得分
        cls_id = dets[:, 5]  # 取出类别 id
        boxes = scale_coords(boxes, ratio, pad)  # 将坐标映射回原图尺度
        return np.concatenate([boxes, scores[:, None], cls_id[:, None]], axis=1)  # 拼接为 [N,6] 返回

    def _postprocess_raw(self, raw, ratio, pad):  # 处理原始输出 [C,Num]
        pred = raw.transpose(1, 0)            # [C,Num] -> [Num,C]
        boxes_xywh = pred[:, :4]              # 取 (cx,cy,w,h)
        cls_scores = pred[:, 4:]              # 取各类别得分
        if cls_scores.size == 0:              # 安全判断：无类别分数
            return np.zeros((0, 6), dtype=np.float32)  # 返回空
        cls_id = np.argmax(cls_scores, axis=1)  # 每个候选的最佳类别索引
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_id]  # 对应最佳分数
        keep = scores >= CONF_THRES           # 依据置信度阈值过滤
        if not np.any(keep):                  # 若没有通过阈值的候选
            return np.zeros((0, 6), dtype=np.float32)  # 返回空
        boxes_xywh = boxes_xywh[keep]; scores = scores[keep]; cls_id = cls_id[keep]  # 过滤后数据
        boxes_xyxy = xywh2xyxy(boxes_xywh)    # 转为 (x1,y1,x2,y2)
        final_boxes, final_scores, final_cls = [], [], []  # 聚合容器
        for c in np.unique(cls_id):           # 按类别分别做 NMS
            idxs = np.where(cls_id == c)[0]   # 当前类别的索引
            b = boxes_xyxy[idxs]; s = scores[idxs]  # 该类别的框和分数
            keep_idx = nms_boxes(b, s, IOU_THRES)   # 执行 NMS 得到保留索引
            final_boxes.append(b[keep_idx])   # 累积保留框
            final_scores.append(s[keep_idx])  # 累积分数
            final_cls.append(np.full((len(keep_idx),), c, dtype=np.float32))  # 累积类别 id
        if len(final_boxes) == 0:             # 若全部为空
            return np.zeros((0, 6), dtype=np.float32)  # 返回空
        boxes = np.concatenate(final_boxes, axis=0)  # 拼接所有框
        scores = np.concatenate(final_scores, axis=0)  # 拼接所有分数
        cls = np.concatenate(final_cls, axis=0)  # 拼接所有类 id
        return np.concatenate([boxes, scores[:, None], cls[:, None]], axis=1).astype(np.float32)  # 返回 [N,6]

# ============================ 简易 IoU 跟踪（给 person 分配稳定 ID） ============================
def iou_xyxy(a, b):  # 计算两个集合框之间的 IoU 矩阵
    tl = np.maximum(a[:, None, :2], b[None, :, :2])  # 交集左上角坐标
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])  # 交集右下角坐标
    wh = np.maximum(0.0, br - tl)  # 交集宽高（小于0置0）
    inter = wh[:, :, 0] * wh[:, :, 1]  # 交集面积
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # A 集合面积
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # B 集合面积
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)  # 返回 IoU 值矩阵

class Track:  # 单条轨迹对象
    def __init__(self, track_id, det):  # 用一个检测初始化轨迹
        self.id = int(track_id)  # 轨迹 ID（唯一）
        self.bbox = det[:4].astype(np.float32)  # 当前边框 xyxy
        self.score = float(det[4])  # 当前置信度
        self.cls_id = int(det[5])  # 类别 ID
        self.lost = 0  # 连续丢失帧数
        self.hits = 1  # 命中帧数
        self.inside = False  # 是否在 ROI 内（上一帧状态）
        self.centroid = self._centroid()  # 当前中心点坐标

    def update(self, det):  # 用新检测更新轨迹
        alpha = 0.7  # 平滑系数（越大越信新值）
        self.bbox = alpha * det[:4].astype(np.float32) + (1 - alpha) * self.bbox  # 指数滑动平均更新 bbox
        self.score = float(det[4])  # 更新分数
        self.hits += 1; self.lost = 0  # 命中+1，丢失清零
        self.centroid = self._centroid()  # 重新计算中心点

    def predict(self):  # 运动预测（占位）
        pass  # 未实现卡尔曼滤波，保持不变

    def _centroid(self):  # 计算 bbox 中心点
        x1, y1, x2, y2 = self.bbox  # 解包坐标
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)  # 返回 (cx, cy)

class IOUTracker:  # 基于 IoU 的简易数据关联跟踪器
    def __init__(self, iou_thres=0.3, max_lost=20, min_box_area=MIN_BOX_AREA):  # 初始化参数
        self.iou_thres = iou_thres  # 匹配阈值
        self.max_lost = max_lost  # 最大允许丢失帧数
        self.min_box_area = min_box_area  # 最小面积阈值
        self.tracks = {}  # 活跃轨迹字典：id -> Track
        self.next_id = 1  # 下一个可用的轨迹 ID

    def update(self, detections):  # 用当前帧检测更新所有轨迹
        dets = detections[detections[:, 5] == 0] if len(detections) > 0 else detections  # 仅保留 person
        if len(dets) > 0:  # 若有候选
            areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])  # 计算面积
            dets = dets[areas >= self.min_box_area]  # 过滤过小的框
        if len(self.tracks) == 0:  # 若没有活跃轨迹
            for det in dets: self._start(det)  # 所有检测都新建轨迹
            return list(self.tracks.values())  # 返回活跃轨迹列表
        for t in self.tracks.values(): t.predict()  # 对既有轨迹做预测（占位）

        if len(dets) > 0:  # 若当前帧有检测
            ids = list(self.tracks.keys())  # 取出轨迹 ID 列表
            trk_boxes = np.array([self.tracks[i].bbox for i in ids], dtype=np.float32)  # 所有轨迹的 bbox
            det_boxes = dets[:, :4].astype(np.float32)  # 当前检测的 bbox
            iou_mat = iou_xyxy(trk_boxes, det_boxes)  # 计算 IoU 矩阵
            used_trk, used_det, pairs = set(), set(), []  # 记录已匹配索引与匹配对
            while True:  # 贪心匹配最大 IoU
                if iou_mat.size == 0: break  # 边界：矩阵为空
                ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)  # 找到最大 IoU 的行列
                if iou_mat[ti, di] < self.iou_thres: break  # 若最大值仍小于阈值，则停止匹配
                if ti in used_trk or di in used_det:  # 已被使用则跳过（理论上不会进此分支）
                    iou_mat[ti, di] = -1; continue  # 将该元素置无效继续
                used_trk.add(ti); used_det.add(di); pairs.append((ti, di))  # 记录匹配
                iou_mat[ti, :] = -1; iou_mat[:, di] = -1  # 屏蔽该行该列，避免重复匹配
            for ti, di in pairs:  # 用匹配到的检测更新对应轨迹
                tid = ids[ti]  # 取轨迹 ID
                self.tracks[tid].update(dets[di])  # 更新轨迹状态
            for di in range(len(dets)):  # 未匹配的检测
                if di not in used_det: self._start(dets[di])  # 启动新轨迹
            for pos, tid in enumerate(ids):  # 未匹配的轨迹
                if pos not in used_trk: self.tracks[tid].lost += 1  # 丢失计数+1
            self._purge()  # 清理超时的轨迹
        else:  # 没有检测
            for t in self.tracks.values(): t.lost += 1  # 所有轨迹丢失+1
            self._purge()  # 清理超时轨迹
        return list(self.tracks.values())  # 返回活跃轨迹列表

    def _start(self, det):  # 启动新轨迹
        t = Track(self.next_id, det)  # 创建 Track 对象
        self.tracks[self.next_id] = t  # 加入字典
        self.next_id += 1  # 下一个可用 ID 自增

    def _purge(self):  # 删除丢失过久的轨迹
        rm = [tid for tid, t in self.tracks.items() if t.lost > self.max_lost]  # 找到需要删除的轨迹 ID
        for tid in rm: del self.tracks[tid]  # 从字典中删除

# ============================ ROI 交互绘制（多边形） ============================
class PolygonDrawer:  # 利用鼠标在窗口里绘制多边形 ROI
    def __init__(self, win_name):  # 构造函数
        self.win = win_name  # 绑定的窗口名称
        self.points = []  # 已点击的顶点集合
        self.closed = False  # 多边形是否闭合
        cv2.setMouseCallback(self.win, self._mouse_cb)  # 注册鼠标事件回调

    def _mouse_cb(self, event, x, y, flags, param):  # 鼠标事件处理
        if self.closed: return  # 若已闭合则不再响应
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下
            self.points.append((x, y))  # 记录一个顶点
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.points) >= 3:  # 右键按下且已有不少于3个点
            self.closed = True  # 标记为闭合完成

    def reset(self):  # 重置多边形（用于重新绘制）
        self.points = []; self.closed = False  # 清空点并标记未闭合

    def draw(self, frame):  # 在帧上绘制当前多边形（点与线）
        for p in self.points:  # 绘制每个顶点
            cv2.circle(frame, p, 4, (0, 255, 255), -1)  # 画小圆点
        for i in range(1, len(self.points)):  # 绘制相邻点之间的线段
            cv2.line(frame, self.points[i-1], self.points[i], (0, 255, 255), 2)  # 连接线
        if self.closed and len(self.points) >= 3:  # 如果闭合
            cv2.line(frame, self.points[-1], self.points[0], (0, 200, 200), 2)  # 闭合首尾

    def mask(self, shape):  # 根据多边形生成布尔掩膜
        m = np.zeros(shape[:2], dtype=np.uint8)  # 创建全零掩膜
        if self.closed and len(self.points) >= 3:  # 若闭合有效
            pts = np.array(self.points, dtype=np.int32)  # 顶点数组
            cv2.fillPoly(m, [pts], 1)  # 填充多边形区域为1
        return m.astype(bool)  # 返回布尔掩膜

    def contains_point(self, pt):  # 判断一个点是否在多边形内（含边界）
        if not (self.closed and len(self.points) >= 3): return False  # 未闭合则直接 False
        pts = np.array(self.points, dtype=np.int32)  # 顶点数组
        return cv2.pointPolygonTest(pts, (float(pt[0]), float(pt[1])), False) >= 0  # >=0 表示在边上或内部

# ============================ 主流程 ============================
def main():  # 主函数：摄像头 -> 推理 -> 跟踪 -> ROI 统计 -> CSV
    cap = cv2.VideoCapture(0)  # 打开默认摄像头（编号 0）
    if not cap.isOpened():  # 若摄像头打开失败
        raise RuntimeError("无法打开摄像头 0，请检查设备与权限。")  # 抛出异常提示

    model = YOLOv8ONNX(ONNX_PATH, use_cuda=USE_CUDA)  # 创建 YOLO 推理器
    print("[INFO] ORT providers:", model.session.get_providers())  # 打印实际使用的执行器列表

    tracker = IOUTracker(iou_thres=0.3, max_lost=20, min_box_area=MIN_BOX_AREA)  # 构建简易 IoU 跟踪器

    csv_file_exists = Path(CSV_PATH).exists()  # 判断 CSV 是否已存在
    csv_f = open(CSV_PATH, mode="a", newline="", encoding="utf-8")  # 以追加方式打开 CSV
    csv_w = csv.writer(csv_f)  # 创建 CSV writer
    if not csv_file_exists:  # 若是新文件
        csv_w.writerow(["timestamp", "event", "track_id", "x1", "y1", "x2", "y2",  # 写入表头
                        "class", "score", "total_entries", "current_inside"])

    win = "YOLOv8-ONNX Person ROI Counter"  # 显示窗口名称
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口
    roi = PolygonDrawer(win)  # 绑定多边形绘制到该窗口

    total_entries = 0  # 累计进入次数（外->内）
    fps = 0.0  # 帧率初值
    color_map = {}  # 轨迹 ID -> 颜色 的映射（稳定视觉识别）

    while True:  # 主循环：逐帧处理
        ok, frame = cap.read()  # 读取一帧图像
        if not ok:  # 若读取失败
            print("[WARN] 读取摄像头失败，继续尝试...")  # 打印警告
            continue  # 跳过本帧

        t0 = time.time()  # 记录开始时间

        roi.draw(frame)  # 绘制当前 ROI（无论是否闭合，先画出）
        if not roi.closed:  # 若 ROI 尚未闭合
            put_text_cn(frame, "左键加点，右键闭合；按 r 重置；按 q 退出",  # 中文提示
                        (10, 20), font_path=FONT_PATH, font_size=FONT_SIZE_SMALL, color=(0,255,255))  # 文本参数
            cv2.imshow(win, frame)  # 显示当前帧
            key = cv2.waitKey(1) & 0xFF  # 监听键盘事件
            if key == ord('r'): roi.reset()  # r 键重置 ROI
            elif key == ord('q'): break  # q 键退出程序
            continue  # 未闭合，不进入检测流程

        dets = model.infer(frame)  # 进行 YOLO 推理，得到 [N,6] 检测结果
        dets = dets[dets[:, 5] == 0] if len(dets) > 0 else dets  # 仅保留 person 类别

        tracks = tracker.update(dets)  # 用检测结果更新跟踪器，获得活跃轨迹列表

        current_inside = 0  # 当前 ROI 内人数计数器
        for t in tracks:  # 遍历每条轨迹
            if t.id not in color_map:  # 为新 ID 分配固定颜色
                color_map[t.id] = tuple(int(c) for c in np.random.randint(0, 255, size=3))  # 随机 BGR
            color = color_map[t.id]  # 取出该 ID 的颜色

            x1, y1, x2, y2 = map(int, t.bbox)  # 将 bbox 转为整数便于绘制
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 画目标矩形框
            put_text_cn(frame, f"ID {t.id} | 行人 {t.score:.2f}",  # 绘制中文标签（ID+类别+分数）
                        (x1, max(0, y1 - 24)), font_path=FONT_PATH, font_size=FONT_SIZE_BIG, color=color)  # 文本样式

            cx, cy = map(int, t.centroid)  # 当前中心点坐标（取整）
            cv2.circle(frame, (cx, cy), 3, color, -1)  # 在中心点画一个小圆
            now_inside = roi.contains_point((cx, cy))  # 判断中心点是否在 ROI 内

            if now_inside: current_inside += 1  # 在内则当前计数+1

            if (not t.inside) and now_inside:  # 触发“进入事件”：外->内
                total_entries += 1  # 累计进入次数+1
                csv_w.writerow([  # 写入一行 CSV 记录
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 时间戳
                    "enter", t.id, x1, y1, x2, y2, "person", f"{t.score:.4f}",  # 事件与目标信息
                    total_entries, current_inside  # 全局累计与当前在内人数
                ])  # CSV 行写完
                csv_f.flush()  # 立即落盘，避免异常退出丢数据
            t.inside = now_inside  # 更新该轨迹的 inside 状态，供下一帧使用

        mask = roi.mask(frame.shape)  # 生成 ROI 区域的布尔掩膜
        overlay = frame.copy()  # 复制一份图像用于混合
        overlay[mask] = (overlay[mask] * 0.5 + np.array([40, 80, 40]) * 0.5).astype(np.uint8)  # ROI 染色
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)  # 叠加半透明效果

        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, time.time() - t0))  # 指数滑动平均计算 FPS

        put_text_cn(frame, f"当前在内人数: {current_inside}", (10, 20),  # 左上角显示当前人数
                    font_path=FONT_PATH, font_size=FONT_SIZE_MED, color=(0,255,255))  # 文本样式
        put_text_cn(frame, f"累计进入次数: {total_entries}", (10, 55),  # 显示累计进入次数
                    font_path=FONT_PATH, font_size=FONT_SIZE_MED, color=(0,220,220))  # 文本样式
        put_text_cn(frame, f"帧率 FPS: {fps:.1f}", (10, 90),  # 显示 FPS
                    font_path=FONT_PATH, font_size=FONT_SIZE_MED, color=(0,200,200))  # 文本样式
        put_text_cn(frame, "按 r 重置ROI；按 q 退出", (10, 125),  # 操作提示
                    font_path=FONT_PATH, font_size=FONT_SIZE_SMALL, color=(0,200,255))  # 文本样式

        roi.draw(frame)  # 再绘制一次 ROI 边线，让其显示在最上层
        cv2.imshow(win, frame)  # 显示最终叠加的图像
        key = cv2.waitKey(1) & 0xFF  # 读取键盘输入
        if key == ord('r'):  # r 键：重置 ROI
            roi.reset()  # 清空多边形
            for t in tracker.tracks.values():  # 将所有轨迹的 inside 状态清零
                t.inside = False  # 避免新 ROI 与旧状态冲突
        elif key == ord('q'):  # q 键：退出
            break  # 跳出循环，收尾

    csv_f.close()  # 关闭 CSV 文件
    cap.release()  # 释放摄像头资源
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

# ============================ 入口 ============================
if __name__ == "__main__":  # 仅当脚本被直接运行时执行
    if not Path(ONNX_PATH).exists():  # 运行前检查模型是否存在
        raise FileNotFoundError(f"未找到 ONNX 模型：{ONNX_PATH}")  # 未找到则抛出异常
    # FONT_PATH = "C:/Windows/Fonts/simhei.ttf"  # 可在此处直接指定字体（示例）
    if FONT_PATH is None and _auto_pick_font() is None:  # 若未设置且自动也找不到
        print("提示：未设置 FONT_PATH，且未能自动找到中文字体；请设置 FONT_PATH 为本机字体路径。")  # 控制台提示
    main()  # 调用主函数，启动程序
