# -*- coding: utf-8 -*-  # 指定源码文件使用 UTF-8 编码，确保中文字符正常显示
# ============================================================
# YOLOv8-ONNX + 摄像头 + 简易目标跟踪（逐行中文注释版）
# 依赖：pip install onnxruntime opencv-python numpy
# 说明：
#   1) 使用 onnxruntime 加载 YOLOv8 的 ONNX 模型进行实时检测
#   2) 使用一个“轻量级 IoU 匹配跟踪器”（无外部依赖）为目标分配稳定的 track_id
#   3) 兼容两种常见 ONNX 导出：A) 已含 NMS 的输出；B) 原始输出(如 [1,84,8400])，本地做解码+NMS
#   4) 若你的模型类别与 COCO80 不同，请修改 CLASS_NAMES
# ============================================================

import time  # 用于计算耗时与 FPS
from pathlib import Path  # 处理跨平台文件路径
import cv2  # OpenCV，用于读写图像、摄像头和可视化
import numpy as np  # 数值计算库
import onnxruntime as ort  # 运行 ONNX 模型的推理引擎

# ============================ 配置区域 ============================
ONNX_PATH = "yolov8n.onnx"  # 指定 ONNX 模型路径（请替换为你的 .onnx 文件）
IMG_SIZE = (640, 640)  # 推理输入分辨率 (width, height)，需与导出一致
CONF_THRES = 0.25  # 置信度阈值（原始输出解码阶段使用）
IOU_THRES = 0.45  # NMS 的 IoU 阈值（原始输出解码阶段使用）
USE_CUDA = True  # 是否优先使用 GPU 推理（需安装 onnxruntime-gpu）
MAX_LOST = 20  # 跟踪器中，轨迹连续丢失多少帧后删除（根据帧率与场景调）
MIN_BOX_AREA = 10 * 10  # 过滤过小目标的面积阈值（像素）
DRAW_TRAIL = True  # 是否绘制跟踪轨迹（历史中心点）

# COCO80 类别名（若自定义数据集，请替换）
CLASS_NAMES = [
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
]  # 以上为 COCO 的 80 类名

# ============================ 通用工具函数（预处理/后处理） ============================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """按比例缩放并填充至目标尺寸，返回处理后图像、缩放比例 r、左上角填充偏移 pad。"""
    shape = im.shape[:2]  # 原图高宽 (h, w)
    if isinstance(new_shape, int):  # 如果传入的是单个整数
        new_shape = (new_shape, new_shape)  # 转为方形 (w, h)
    new_w, new_h = new_shape  # 目标宽高

    r = min(new_w / shape[1], new_h / shape[0])  # 计算缩放比例，保证不超边
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # 缩放后未填充的尺寸
    dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]  # 需要填充的尺寸差
    dw /= 2  # 左右均分
    dh /= 2  # 上下均分

    if shape[::-1] != new_unpad:  # 如果需要缩放
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)  # 双线性插值缩放
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上/下填充
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左/右填充
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 边缘填充
    return im, r, (left, top)  # 返回处理后图像、缩放比例、左上填充

def nms_boxes(boxes, scores, iou_thres=0.5):
    """基于 NumPy 的 NMS（非极大值抑制），传入 xyxy 框与分数，返回保留索引。"""
    x1, y1, x2, y2 = boxes.T  # 拆分坐标
    areas = (x2 - x1) * (y2 - y1)  # 框面积
    order = scores.argsort()[::-1]  # 分数从大到小排序索引
    keep = []  # 记录保留的索引
    while order.size > 0:  # 直到没有候选
        i = order[0]  # 当前分数最高的索引
        keep.append(i)  # 保留该索引
        if order.size == 1:  # 若只剩这一个
            break  # 结束循环
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 交集左上 x
        yy1 = np.maximum(y1[i], y1[order[1:]])  # 交集左上 y
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 交集右下 x
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 交集右下 y
        w = np.maximum(0.0, xx2 - xx1)  # 交集宽
        h = np.maximum(0.0, yy2 - yy1)  # 交集高
        inter = w * h  # 交集面积
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)  # IoU 值
        inds = np.where(iou <= iou_thres)[0]  # IoU 小于阈值的保留
        order = order[inds + 1]  # 更新剩余候选索引集合
    return keep  # 返回保留的索引

def scale_coords(boxes, ratio, pad):
    """将 letterbox 尺度的 xyxy 坐标映射回原图尺度。"""
    boxes[:, [0, 2]] -= pad[0]  # 去除 x 方向的左侧填充
    boxes[:, [1, 3]] -= pad[1]  # 去除 y 方向的上侧填充
    boxes[:, :4] /= ratio  # 除以缩放比例回到原图尺寸
    return boxes  # 返回映射后的坐标

def xywh2xyxy(xywh):
    """将 (cx,cy,w,h) 转为 (x1,y1,x2,y2)。"""
    xyxy = np.zeros_like(xywh)  # 创建同形状容器
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2
    return xyxy  # 返回 xyxy

# ============================ ONNX 推理包装 ============================
class YOLOv8ONNX:
    """封装 YOLOv8-ONNX 推理，统一输出为 [N,6]: (x1,y1,x2,y2,score,cls) 的原图尺度结果。"""
    def __init__(self, onnx_path, use_cuda=True):
        """构造函数：创建 onnxruntime 会话，记录输入/输出信息。"""
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]  # 根据配置选择提供者
        try:  # 尝试使用指定 providers 创建推理会话
            self.session = ort.InferenceSession(Path(onnx_path).as_posix(), providers=providers)  # 创建会话
        except Exception:  # 若失败（如未安装 GPU 版 ORT）
            self.session = ort.InferenceSession(Path(onnx_path).as_posix(), providers=["CPUExecutionProvider"])  # 回退 CPU
        self.input_name = self.session.get_inputs()[0].name  # 记录第一个输入的名称
        self.outputs_info = [(o.name, o.shape) for o in self.session.get_outputs()]  # 记录所有输出的名称与形状（调试/判断格式用）

    def infer(self, img_bgr):
        """对 BGR 图像进行推理，返回 [N,6] 原图尺度的检测结果。"""
        img0 = img_bgr  # 保存原始图像引用
        lb_img, ratio, pad = letterbox(img0, (IMG_SIZE[0], IMG_SIZE[1]))  # 预处理：letterbox 缩放+填充
        img_rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        img = img_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # HWC->CHW，添加 batch 维度，归一化到 [0,1]

        outputs = self.session.run(None, {self.input_name: img})  # 执行推理，取所有输出

        dets = None  # 初始化最终检测结果
        if len(outputs) == 1:  # 若只有一个输出
            out = outputs[0]  # 取该输出
            if out.ndim == 2 and out.shape[1] in (6, 7):  # 情况 A：已含 NMS 的二维结果
                if out.shape[1] == 7:  # 若多出 batch_id 列
                    out = out[:, 1:]  # 去掉 batch_id
                dets = out  # 直接作为检测结果
            elif out.ndim == 3 and out.shape[-1] in (6, 7):  # 情况 A：形如 [1,N,6/7]
                out = out[0]  # 降维去掉 batch 维度
                if out.shape[1] == 7:  # 同样去掉 batch_id 列
                    out = out[:, 1:]
                dets = out  # 作为检测结果
            elif out.ndim == 3 and out.shape[0] == 1 and out.shape[1] >= 5:  # 情况 B：原始输出 [1,84,8400] 等
                dets = self._postprocess_raw(out[0], ratio, pad)  # 进行本地解码+NMS，得到 letterbox 尺度结果
        else:  # 若有多个输出（某些导出版本可能如此）
            det = None  # 临时结果
            for o in outputs:  # 遍历各输出
                if o.ndim == 2 and o.shape[1] in (6, 7):  # 优先尝试已含 NMS 的二维输出
                    det = o if o.shape[1] == 6 else o[:, 1:]  # 处理 batch_id
                    break  # 找到即止
                if o.ndim == 3 and o.shape[-1] in (6, 7):  # 或三维 [1,N,6/7]
                    det = o[0] if o.shape[-1] == 6 else o[0][:, 1:]  # 去掉 batch 维/列
                    break  # 找到即止
            if det is not None:  # 若找到已含 NMS 的结果
                dets = det  # 直接使用
            else:  # 否则尝试原始输出
                for o in outputs:  # 遍历查找原始形状
                    if o.ndim == 3 and o.shape[0] == 1 and o.shape[1] >= 5:  # 匹配原始输出
                        dets = self._postprocess_raw(o[0], ratio, pad)  # 本地解码
                        break  # 完成即止

        if dets is None or dets.size == 0:  # 若无检测结果
            return np.zeros((0, 6), dtype=np.float32)  # 返回空数组

        boxes = dets[:, :4].copy()  # 拷贝 xyxy
        scores = dets[:, 4]  # 取 score
        cls_id = dets[:, 5]  # 取 cls
        boxes = scale_coords(boxes, ratio, pad)  # 映射回原图尺度
        dets_final = np.concatenate([boxes, scores[:, None], cls_id[:, None]], axis=1)  # 拼回 [N,6]
        return dets_final  # 返回最终检测结果（原图尺度）

    def _postprocess_raw(self, raw, ratio, pad):
        """处理原始输出 (C,Num) -> [N,6]（在 letterbox 尺度），包括取最优类别、阈值过滤与 NMS。"""
        pred = raw.transpose(1, 0)  # 转为 (Num, C)，如 [8400,84]
        boxes_xywh = pred[:, :4]  # 取 (cx,cy,w,h)
        cls_scores = pred[:, 4:]  # 取各类别分数
        if cls_scores.size == 0:  # 安全检查
            return np.zeros((0, 6), dtype=np.float32)  # 返回空

        cls_id = np.argmax(cls_scores, axis=1)  # 每个候选框选择分数最高的类别
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_id]  # 对应的分数即该框置信度
        keep = scores >= CONF_THRES  # 置信度阈值过滤
        if not np.any(keep):  # 若无通过者
            return np.zeros((0, 6), dtype=np.float32)  # 返回空

        boxes_xywh = boxes_xywh[keep]  # 过滤后框
        scores = scores[keep]  # 过滤后分数
        cls_id = cls_id[keep]  # 过滤后类别
        boxes_xyxy = xywh2xyxy(boxes_xywh)  # 坐标格式转换便于 NMS

        final_boxes, final_scores, final_cls = [], [], []  # 存储各类别 NMS 后结果
        for c in np.unique(cls_id):  # 对每个类别分别做 NMS（避免跨类互相抑制）
            idxs = np.where(cls_id == c)[0]  # 找到该类的索引
            b = boxes_xyxy[idxs]  # 取该类框
            s = scores[idxs]  # 取该类分数
            keep_idx = nms_boxes(b, s, IOU_THRES)  # NMS 过滤得到保留索引
            final_boxes.append(b[keep_idx])  # 保存保留框
            final_scores.append(s[keep_idx])  # 保存保留分数
            final_cls.append(np.full((len(keep_idx),), c, dtype=np.float32))  # 保存保留类别

        if len(final_boxes) == 0:  # 若所有类别都空
            return np.zeros((0, 6), dtype=np.float32)  # 返回空

        boxes = np.concatenate(final_boxes, axis=0)  # 拼接所有类别的框
        scores = np.concatenate(final_scores, axis=0)  # 拼接所有类别分数
        cls = np.concatenate(final_cls, axis=0)  # 拼接所有类别 id
        return np.concatenate([boxes, scores[:, None], cls[:, None]], axis=1).astype(np.float32)  # 返回 [N,6]

# ============================ 简易 IoU 跟踪器 ============================
def iou_xyxy(a, b):
    """计算两个 xyxy 框的 IoU（支持批量：a[N,4], b[M,4]）。"""
    # 扩展维度使用广播，计算交集坐标
    tl = np.maximum(a[:, None, :2], b[None, :, :2])  # 左上角 (max of x1,y1)
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])  # 右下角 (min of x2,y2)
    wh = np.maximum(0.0, br - tl)  # 交集宽高
    inter = wh[:, :, 0] * wh[:, :, 1]  # 交集面积
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # a 的面积
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # b 的面积
    iou = inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)  # IoU = 交/并
    return iou  # 返回 IoU 矩阵 [Na, Nb]

class Track:
    """单个轨迹对象，保存边框、类别、分数、ID、丢失计数、轨迹点等信息。"""
    def __init__(self, track_id, bbox, score, cls_id):
        """构造函数：用检测框初始化轨迹。"""
        self.id = track_id  # 轨迹 ID（全局唯一）
        self.bbox = bbox.astype(np.float32)  # 当前边框 xyxy（float）
        self.score = float(score)  # 当前置信度
        self.cls_id = int(cls_id)  # 类别 id
        self.lost = 0  # 连续丢失帧数
        self.hits = 1  # 命中次数（用于初期稳定）
        self.trail = []  # 历史中心点列表（用于画轨迹）
        self.update_trail()  # 初始化时记录一次中心点

    def update(self, bbox, score):
        """用新的检测结果更新轨迹；可选做平滑。"""
        # 这里做一个简易平滑：新框与旧框做指数滑动平均（可调 alpha）
        alpha = 0.7  # 平滑系数（越大越信新值）
        self.bbox = alpha * bbox.astype(np.float32) + (1 - alpha) * self.bbox  # 平滑更新 bbox
        self.score = float(score)  # 更新分数
        self.lost = 0  # 命中，丢失计数清零
        self.hits += 1  # 命中次数+1
        self.update_trail()  # 记录中心点

    def predict(self):
        """预测步（占位）：如需卡尔曼滤波可在此更新；当前简化为不动。"""
        # 简易版不做运动模型预测，保持 bbox 不变
        pass  # 保持原样

    def update_trail(self):
        """将当前中心点加入轨迹列表，长度做裁剪防止过长。"""
        x1, y1, x2, y2 = self.bbox  # 拆出坐标
        cx = int((x1 + x2) / 2)  # 中心 x
        cy = int((y1 + y2) / 2)  # 中心 y
        self.trail.append((cx, cy))  # 追加中心点
        if len(self.trail) > 30:  # 轨迹长度限制（防止过长）
            self.trail.pop(0)  # 超限则弹出最早记录

class IOUTracker:
    """基于 IoU 的简单数据关联跟踪器（贪心匹配），适合演示/入门。"""
    def __init__(self, max_lost=30, iou_match_thres=0.3, min_box_area=100):
        """构造函数：配置丢失阈值、匹配阈值与最小面积过滤。"""
        self.max_lost = max_lost  # 最大允许连续丢失帧数
        self.iou_match_thres = iou_match_thres  # 匹配 IoU 阈值
        self.min_box_area = min_box_area  # 最小面积阈值
        self.tracks = {}  # 活跃轨迹字典：id -> Track
        self.next_id = 1  # 分配下一个轨迹 ID 的计数器

    def update(self, detections):
        """
        用当前帧的检测结果更新所有轨迹。
        detections: np.ndarray [N,6] -> (x1,y1,x2,y2,score,cls_id)
        返回：当前活跃轨迹列表（Track 对象）
        """
        # 过滤过小框（避免噪声）
        if len(detections) > 0:  # 若有检测
            wh = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])  # 计算面积
            detections = detections[wh >= self.min_box_area]  # 保留面积达标的检测
        # 若当前无活跃轨迹，直接将所有检测初始化为新轨迹
        if len(self.tracks) == 0:  # 没有活跃轨迹
            for det in detections:  # 遍历检测
                self._start_track(det)  # 启动新轨迹
            return list(self.tracks.values())  # 返回当前轨迹

        # 1) 先对既有轨迹做“预测步”（这里占位不动，若用卡尔曼可更新）
        for t in self.tracks.values():  # 遍历活跃轨迹
            t.predict()  # 预测（占位）

        # 2) 若有检测，则计算 IoU 进行匹配
        if len(detections) > 0:  # 有检测时
            track_ids = list(self.tracks.keys())  # 取出所有轨迹 id
            track_boxes = np.array([self.tracks[i].bbox for i in track_ids], dtype=np.float32)  # 收集轨迹的 bbox
            det_boxes = detections[:, :4].astype(np.float32)  # 当前检测的 bbox
            iou_mat = iou_xyxy(track_boxes, det_boxes)  # 计算轨迹与检测之间的 IoU 矩阵
            # 贪心匹配：每次选择当前最大 IoU 的轨迹-检测对（大于阈值）
            matched_trk, matched_det = set(), set()  # 记录已经匹配的轨迹与检测索引
            pairs = []  # 保存匹配对 (trk_idx, det_idx)
            while True:  # 迭代寻找最大 IoU
                if iou_mat.size == 0:  # 边界情况：无元素
                    break  # 退出循环
                trk_idx, det_idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)  # 找到 IoU 最大的位置
                if iou_mat[trk_idx, det_idx] < self.iou_match_thres:  # 若最大 IoU 仍小于匹配阈值
                    break  # 无法继续匹配
                if trk_idx in matched_trk or det_idx in matched_det:  # 若已匹配过则跳过（但理论上不会发生）
                    iou_mat[trk_idx, det_idx] = -1  # 标记为已用
                    continue  # 继续查找
                matched_trk.add(trk_idx)  # 标记该轨迹已匹配
                matched_det.add(det_idx)  # 标记该检测已匹配
                pairs.append((trk_idx, det_idx))  # 记录匹配对
                iou_mat[trk_idx, :] = -1  # 将该轨迹所在行置无效（-1）
                iou_mat[:, det_idx] = -1  # 将该检测所在列置无效（-1）

            # 3) 用匹配到的检测更新相应轨迹
            for trk_idx, det_idx in pairs:  # 遍历匹配对
                trk_id = track_ids[trk_idx]  # 取对应轨迹 id
                det = detections[det_idx]  # 取对应检测
                self.tracks[trk_id].update(det[:4], det[4])  # 更新该轨迹的 bbox 和 score（类别一般保持首次，检测可选更新）
                # 可选：若类别变化明显，也可以更新 cls_id，这里保持初始类别更稳定

            # 4) 对未匹配到的检测，启动新轨迹
            for det_idx in range(len(detections)):  # 遍历所有检测索引
                if det_idx not in matched_det:  # 若该检测未被匹配
                    self._start_track(detections[det_idx])  # 新建轨迹

            # 5) 对未匹配到的轨迹，增加丢失计数；超过阈值则删除
            for trk_pos, trk_id in enumerate(track_ids):  # 遍历旧轨迹
                if trk_pos not in matched_trk:  # 若该旧轨迹未匹配
                    self.tracks[trk_id].lost += 1  # 丢失计数 +1
            self._purge_tracks()  # 清理过期轨迹
        else:
            # 没有检测时，所有轨迹丢失计数 +1
            for trk in self.tracks.values():  # 遍历活跃轨迹
                trk.lost += 1  # 丢失计数 +1
            self._purge_tracks()  # 清理过期轨迹

        return list(self.tracks.values())  # 返回当前仍然活跃的轨迹列表

    def _start_track(self, det):
        """根据单个检测 (x1,y1,x2,y2,score,cls) 启动一条新轨迹。"""
        x1, y1, x2, y2, s, c = det  # 拆出检测信息
        track = Track(self.next_id, np.array([x1, y1, x2, y2], dtype=np.float32), s, c)  # 创建轨迹对象
        self.tracks[self.next_id] = track  # 加入活跃字典
        self.next_id += 1  # 轨迹 ID 递增

    def _purge_tracks(self):
        """删除连续丢失超过阈值的轨迹。"""
        to_del = [tid for tid, t in self.tracks.items() if t.lost > self.max_lost]  # 找出超限轨迹 id
        for tid in to_del:  # 遍历需删除的 id
            del self.tracks[tid]  # 从字典中删除该轨迹

# ============================ 主流程：摄像头 + 检测 + 跟踪 ============================
def main():
    """主函数：打开摄像头 -> YOLOv8-ONNX 检测 -> IoU 跟踪 -> 可视化。"""
    cap = cv2.VideoCapture(0)  # 打开默认摄像头（索引 0），如有多路可改 1/2/...
    if not cap.isOpened():  # 检查摄像头是否成功打开
        raise RuntimeError("无法打开摄像头 0，请检查设备连接与权限。")  # 抛出异常提醒

    model = YOLOv8ONNX(ONNX_PATH, use_cuda=USE_CUDA)  # 构建 ONNX 模型推理器
    print("[INFO] ONNX providers:", model.session.get_providers())  # 打印实际使用的 providers（便于确认是否启用 GPU）

    tracker = IOUTracker(max_lost=MAX_LOST, iou_match_thres=0.3, min_box_area=MIN_BOX_AREA)  # 创建 IoU 跟踪器实例
    fps = 0.0  # 初始化 FPS 值
    color_map = {}  # 为每个 track_id 分配一种颜色（可视化更直观）

    while True:  # 主循环：逐帧处理
        ok, frame = cap.read()  # 从摄像头读取一帧
        if not ok:  # 若读取失败
            print("[WARN] 读取摄像头帧失败，继续尝试...")  # 打印警告
            continue  # 跳过本帧

        t0 = time.time()  # 记录起始时间
        detections = model.infer(frame)  # 执行检测，得到 [N,6] -> (x1,y1,x2,y2,score,cls)
        tracks = tracker.update(detections)  # 用检测结果更新跟踪器，得到活跃轨迹列表
        t1 = time.time()  # 记录结束时间
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, t1 - t0))  # 计算指数滑动平均 FPS，平滑稳定

        # 遍历活跃轨迹，绘制框与标签
        for t in tracks:  # t 为 Track 对象
            x1, y1, x2, y2 = map(int, t.bbox)  # 转为整数坐标
            # 为每个 track_id 分配并缓存一个固定颜色（便于跨帧识别）
            if t.id not in color_map:  # 若该 ID 还未分配颜色
                color_map[t.id] = tuple(int(c) for c in np.random.randint(0, 255, size=3))  # 随机生成 BGR 颜色
            color = color_map[t.id]  # 取出该 ID 的颜色

            # 准备标签文本：ID、类别名、置信度
            cls_name = CLASS_NAMES[t.cls_id] if 0 <= t.cls_id < len(CLASS_NAMES) else str(t.cls_id)  # 类别名或 id
            label = f"ID {t.id} | {cls_name} {t.score:.2f}"  # 拼接显示文本

            # 绘制边框与标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 画矩形框
            cv2.putText(frame, label, (x1, max(0, y1 - 7)),  # 在框上方绘制文本标签
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)  # 指定字体、大小、颜色、线宽、抗锯齿

            # 可选：绘制轨迹历史点
            if DRAW_TRAIL and len(t.trail) >= 2:  # 若启用轨迹且点数足够
                for i in range(1, len(t.trail)):  # 遍历相邻历史点
                    cv2.line(frame, t.trail[i - 1], t.trail[i], color, 2)  # 画线连接相邻点

        # 在屏幕左上角绘制 FPS 信息
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),  # 文本位置
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)  # 字体、字号、颜色、线宽、抗锯齿

        cv2.imshow("YOLOv8-ONNX Webcam Tracking", frame)  # 显示结果窗口
        key = cv2.waitKey(1) & 0xFF  # 等待键盘输入（1 ms）
        if key == ord('q'):  # 若按下 'q'，退出主循环
            break  # 跳出循环

    cap.release()  # 释放摄像头资源
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

# ============================ 程序入口 ============================
if __name__ == "__main__":  # 当脚本作为主程序执行时
    if not Path(ONNX_PATH).exists():  # 运行前检查模型文件是否存在
        raise FileNotFoundError(f"未找到 ONNX 模型：{ONNX_PATH}，请修改 ONNX_PATH 为你的模型路径。")  # 若不存在则报错
    main()  # 调用主函数，开始摄像头检测与跟踪
