# -*- coding: utf-8 -*-  # 指定源码文件编码为 UTF-8，确保中文注释与字符串正常显示
# ============================================================
# 零售与运营洞察 Demo（客流热力图 / 停留时长 / 动线分析 / 货架到访 & 触达率）
# - 基于 YOLOv8 (ONNX) + 摄像头 + 简易 IoU 跟踪
# - 多多边形 ROI（可用来标注货架/区域），交互绘制完成后开始统计
# - 实时：检测→跟踪→热力图→轨迹→停留时长→到访事件→触达率
# 依赖安装：
#   pip install onnxruntime opencv-python numpy pillow
#   # 若有 NVIDIA GPU 想用 GPU 推理：
#   # pip uninstall -y onnxruntime && pip install onnxruntime-gpu
# ============================================================

import time  # 计时/FPS/停留时长计算
from datetime import datetime  # 事件时间戳
from pathlib import Path  # 跨平台路径处理
import csv  # 写入事件/统计 CSV
import cv2  # OpenCV：视频、绘制、交互
import numpy as np  # 矩阵/数值运算
import onnxruntime as ort  # ONNX 模型推理
from PIL import Image, ImageDraw, ImageFont  # Pillow：正确渲染中文文本

# ============================ 基本配置 ============================
ONNX_PATH = "yolov8n.onnx"  # 模型路径（替换为你的 yolov8*.onnx）
IMG_SIZE = (640, 640)  # 输入分辨率，需与导出一致 (w, h)
CONF_THRES = 0.25  # 置信度阈值（原始输出解码用）
IOU_THRES = 0.45  # NMS IoU 阈值（原始输出解码用）
USE_CUDA = True  # True 优先 CUDA（需 onnxruntime-gpu）
CLASS_PERSON = 0  # COCO 中 person 类别 id
HEATMAP_DOWNSAMPLE = 4  # 热力图下采样倍率（越大越省内存，越平滑）
HEATMAP_ALPHA = 0.45  # 热力图叠加透明度
TRAIL_MAXLEN = 48  # 轨迹点最大保留数量
EVENTS_CSV = "retail_events.csv"  # 事件日志 CSV
DWELL_SUMMARY_CSV = "retail_dwell_summary.csv"  # 停留时长汇总 CSV（退出时写）
HEATMAP_SNAPSHOT = "retail_heatmap.png"  # 保存热力图截图路径
FONT_PATH = None  # 中文字体路径（None 将自动探测；建议手动设定以确保可用）
SHELF_NAME_PREFIX = "货架"  # ROI 默认前缀名
# ============================ COCO 类别名（仅用到 person） ============================
COCO_CLASSES = [
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
]  # 80 类列表

# ============================ 中文文本渲染（Pillow） ============================
def _auto_pick_font():  # 自动探测常见中文字体
    """返回一个可用的中文字体路径；找不到返回 None。"""  # 文档字符串
    candidates = [  # 常见平台字体路径
        "C:/Windows/Fonts/simhei.ttf",  # Win 黑体
        "C:/Windows/Fonts/msyh.ttc",  # Win 微软雅黑
        "/System/Library/Fonts/PingFang.ttc",  # macOS 苹方
        "/System/Library/Fonts/STHeiti Medium.ttc",  # macOS 华文黑体
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux Noto
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux Noto
    ]  # 结束
    for p in candidates:  # 遍历候选
        if Path(p).exists():  # 若存在
            return p  # 返回该路径
    return None  # 否则返回 None

def put_text_cn(img_bgr, text, org, font_path=None, font_size=22, color=(0,255,255), bold=True):  # 中文渲染
    """在 OpenCV 图像上正确绘制中文文本，支持描边。"""  # 文档
    if font_path is None:  # 若未指定字体
        font_path = _auto_pick_font()  # 自动探测
    if font_path is None or not Path(font_path).exists():  # 若仍不可用
        raise FileNotFoundError("未找到中文字体，请设置 FONT_PATH 为本机字体路径。")  # 抛错提示
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))  # BGR→RGB→PIL
    draw = ImageDraw.Draw(img_pil)  # 获取绘图对象
    font = ImageFont.truetype(font_path, font_size)  # 加载字体
    x, y = org  # 解包位置
    if bold:  # 若需要描边
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:  # 八邻域
            draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0))  # 黑色描边
    draw.text((x, y), text, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))  # 正文（RGB）
    img_bgr[:, :, :] = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)  # 回写 BGR
    return img_bgr  # 返回引用

# ============================ YOLOv8 ONNX 推理（兼容含/不含 NMS 导出） ============================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):  # 等比缩放+填充
    shape = im.shape[:2]  # h,w
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)  # 单值→方形
    new_w, new_h = new_shape  # 目标尺寸
    r = min(new_w / shape[1], new_h / shape[0])  # 缩放比例
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # 缩放后尺寸
    dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]  # 剩余用于填充
    dw /= 2; dh /= 2  # 均分
    if shape[::-1] != new_unpad:  # 若需缩放
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)  # 双线性缩放
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 上下填充
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 左右填充
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 常数色
    return im, r, (left, top)  # 返回处理图、比例、pad

def nms_boxes(boxes, scores, iou_thres=0.5):  # 纯 NumPy NMS
    x1, y1, x2, y2 = boxes.T  # 拆坐标
    areas = (x2 - x1) * (y2 - y1)  # 面积
    order = scores.argsort()[::-1]  # 分数降序索引
    keep = []  # 保留索引
    while order.size > 0:  # 循环
        i = order[0]  # 最高分
        keep.append(i)  # 保留
        if order.size == 1: break  # 结束
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 交集左上x
        yy1 = np.maximum(y1[i], y1[order[1:]])  # 交集左上y
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 交集右下x
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 交集右下y
        w = np.maximum(0.0, xx2 - xx1)  # 宽
        h = np.maximum(0.0, yy2 - yy1)  # 高
        inter = w * h  # 交集面积
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)  # IoU
        inds = np.where(iou <= iou_thres)[0]  # 保留的索引（IoU小于阈值）
        order = order[inds + 1]  # 更新序列
    return keep  # 返回保留

def scale_coords(boxes, ratio, pad):  # 将 letterbox 坐标映射回原图
    boxes[:, [0, 2]] -= pad[0]  # 去 x pad
    boxes[:, [1, 3]] -= pad[1]  # 去 y pad
    boxes[:, :4] /= ratio  # 除以缩放比
    return boxes  # 返回

def xywh2xyxy(xywh):  # (cx,cy,w,h)→(x1,y1,x2,y2)
    xyxy = np.zeros_like(xywh)  # 预分配
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2
    return xyxy  # 返回

class YOLOv8ONNX:  # 模型封装
    """输出统一为 [N,6]: x1,y1,x2,y2,score,cls（原图尺度）。"""  # 说明
    def __init__(self, onnx_path, use_cuda=True):  # 构造
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]  # 执行器
        try:  # 尝试 GPU
            self.session = ort.InferenceSession(Path(onnx_path).as_posix(), providers=providers)  # 创建会话
        except Exception:  # 失败回退 CPU
            self.session = ort.InferenceSession(Path(onnx_path).as_posix(), providers=["CPUExecutionProvider"])  # CPU
        self.input_name = self.session.get_inputs()[0].name  # 输入名
        self.outputs_info = [(o.name, o.shape) for o in self.session.get_outputs()]  # 输出信息

    def infer(self, img_bgr):  # 前向推理
        img0 = img_bgr  # 原图
        lb_img, ratio, pad = letterbox(img0, (IMG_SIZE[0], IMG_SIZE[1]))  # letterbox
        img_rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)  # BGR->RGB
        img = img_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # HWC->CHW+batch+归一化
        outputs = self.session.run(None, {self.input_name: img})  # 推理

        dets = None  # 结果占位
        if len(outputs) == 1:  # 单输出
            out = outputs[0]  # 取出
            if out.ndim == 2 and out.shape[1] in (6, 7):  # 已含 NMS
                if out.shape[1] == 7: out = out[:, 1:]  # 去 batch_id
                dets = out  # 赋值
            elif out.ndim == 3 and out.shape[-1] in (6, 7):  # [1,N,6/7]
                out = out[0]  # 去 batch 维
                if out.shape[1] == 7: out = out[:, 1:]  # 去 batch_id
                dets = out  # 赋值
            elif out.ndim == 3 and out.shape[0] == 1 and out.shape[1] >= 5:  # 原始输出
                dets = self._postprocess_raw(out[0], ratio, pad)  # 本地解码
        else:  # 多输出
            det = None  # 占位
            for o in outputs:  # 查找已含 NMS 的
                if o.ndim == 2 and o.shape[1] in (6, 7):
                    det = o if o.shape[1] == 6 else o[:, 1:]; break  # 找到退出
                if o.ndim == 3 and o.shape[-1] in (6, 7):
                    det = o[0] if o.shape[-1] == 6 else o[0][:, 1:]; break  # 找到退出
            if det is not None:  # 若找到
                dets = det  # 赋值
            else:  # 否则解码原始
                for o in outputs:
                    if o.ndim == 3 and o.shape[0] == 1 and o.shape[1] >= 5:
                        dets = self._postprocess_raw(o[0], ratio, pad); break  # 解码

        if dets is None or dets.size == 0:  # 无结果
            return np.zeros((0, 6), dtype=np.float32)  # 返回空
        boxes = dets[:, :4].copy()  # 坐标
        scores = dets[:, 4]  # 分数
        cls_id = dets[:, 5]  # 类别
        boxes = scale_coords(boxes, ratio, pad)  # 映射回原图
        return np.concatenate([boxes, scores[:, None], cls_id[:, None]], axis=1)  # [N,6]

    def _postprocess_raw(self, raw, ratio, pad):  # 原始输出解码
        pred = raw.transpose(1, 0)  # [C,Num]→[Num,C]
        boxes_xywh = pred[:, :4]  # 取框
        cls_scores = pred[:, 4:]  # 取分
        if cls_scores.size == 0:  # 边界
            return np.zeros((0, 6), dtype=np.float32)  # 空
        cls_id = np.argmax(cls_scores, axis=1)  # 最优类
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_id]  # 对应分
        keep = scores >= CONF_THRES  # 阈值过滤
        if not np.any(keep):  # 如果全丢
            return np.zeros((0, 6), dtype=np.float32)  # 空
        boxes_xywh = boxes_xywh[keep]; scores = scores[keep]; cls_id = cls_id[keep]  # 过滤
        boxes_xyxy = xywh2xyxy(boxes_xywh)  # 转 xyxy
        final_boxes, final_scores, final_cls = [], [], []  # 聚合
        for c in np.unique(cls_id):  # 分类 NMS
            idxs = np.where(cls_id == c)[0]  # 索引
            b = boxes_xyxy[idxs]; s = scores[idxs]  # 框/分
            keep_idx = nms_boxes(b, s, IOU_THRES)  # NMS
            final_boxes.append(b[keep_idx]); final_scores.append(s[keep_idx]); final_cls.append(np.full((len(keep_idx),), c, dtype=np.float32))  # 累积
        if len(final_boxes) == 0:  # 边界
            return np.zeros((0, 6), dtype=np.float32)  # 空
        boxes = np.concatenate(final_boxes, axis=0); scores = np.concatenate(final_scores, axis=0); cls = np.concatenate(final_cls, axis=0)  # 拼接
        return np.concatenate([boxes, scores[:, None], cls[:, None]], axis=1).astype(np.float32)  # [N,6]

# ============================ 简易 IoU 跟踪（含轨迹点） ============================
def iou_xyxy(a, b):  # 计算 IoU 矩阵
    tl = np.maximum(a[:, None, :2], b[None, :, :2])  # 左上
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])  # 右下
    wh = np.maximum(0.0, br - tl)  # 宽高
    inter = wh[:, :, 0] * wh[:, :, 1]  # 交集
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # 面积A
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # 面积B
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)  # IoU

class Track:  # 轨迹
    def __init__(self, track_id, det):  # 初始化
        self.id = int(track_id)  # ID
        self.bbox = det[:4].astype(np.float32)  # 边框
        self.score = float(det[4])  # 分数
        self.cls_id = int(det[5])  # 类别
        self.lost = 0  # 丢失计数
        self.trail = []  # 轨迹点
        self._update_trail()  # 初次记录
    def update(self, det):  # 更新
        alpha = 0.7  # 平滑
        self.bbox = alpha * det[:4].astype(np.float32) + (1 - alpha) * self.bbox  # 平滑坐标
        self.score = float(det[4])  # 更新分
        self.lost = 0  # 命中清零
        self._update_trail()  # 追加轨迹点
    def predict(self):  # 预测（占位）
        pass  # 简化，不做卡尔曼
    def centroid(self):  # 中心点
        x1,y1,x2,y2 = self.bbox  # 解包
        return int((x1+x2)/2), int((y1+y2)/2)  # 返回 (cx,cy)
    def _update_trail(self):  # 轨迹维护
        self.trail.append(self.centroid())  # 加入当前中心
        if len(self.trail) > TRAIL_MAXLEN: self.trail.pop(0)  # 超长裁剪

class IOUTracker:  # 跟踪器
    def __init__(self, iou_thres=0.3, max_lost=20, min_area=15*15):  # 构造
        self.iou_thres = iou_thres  # 匹配阈值
        self.max_lost = max_lost  # 最大丢失帧
        self.min_area = min_area  # 最小面积过滤
        self.tracks = {}  # 活跃轨迹 id->Track
        self.next_id = 1  # 下一个ID
    def update(self, detections):  # 用检测更新
        dets = detections[detections[:,5]==CLASS_PERSON] if len(detections)>0 else detections  # 仅 person
        if len(dets)>0:  # 面积过滤
            areas=(dets[:,2]-dets[:,0])*(dets[:,3]-dets[:,1]); dets=dets[areas>=self.min_area]  # 过滤
        if len(self.tracks)==0:  # 无轨迹
            for d in dets: self._start(d)  # 全部建轨
            return list(self.tracks.values())  # 返回
        for t in self.tracks.values(): t.predict()  # 预测步（占位）
        if len(dets)>0:  # 有检测
            ids=list(self.tracks.keys())  # 轨迹ID列表
            tboxes=np.array([self.tracks[i].bbox for i in ids],dtype=np.float32)  # 轨迹框
            dboxes=dets[:,:4].astype(np.float32)  # 检测框
            iou_mat=iou_xyxy(tboxes, dboxes)  # IoU 矩阵
            used_t, used_d, pairs=set(), set(), []  # 匹配簿
            while True:  # 贪心匹配
                if iou_mat.size==0: break  # 空
                ti,di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)  # 最大值索引
                if iou_mat[ti,di] < self.iou_thres: break  # 小于阈值停止
                if ti in used_t or di in used_d: iou_mat[ti,di]=-1; continue  # 已用则跳过
                used_t.add(ti); used_d.add(di); pairs.append((ti,di))  # 记录配对
                iou_mat[ti,:]=-1; iou_mat[:,di]=-1  # 屏蔽行列
            for ti,di in pairs:  # 用配对更新轨迹
                tid=ids[ti]; self.tracks[tid].update(dets[di])  # 更新
            for di in range(len(dets)):  # 未匹配检测
                if di not in used_d: self._start(dets[di])  # 新建轨迹
            for pos,tid in enumerate(ids):  # 未匹配轨迹
                if pos not in used_t: self.tracks[tid].lost += 1  # 丢失+1
            self._purge()  # 清理过期轨迹
        else:  # 无检测
            for t in self.tracks.values(): t.lost += 1  # 全部丢失+1
            self._purge()  # 清理
        return list(self.tracks.values())  # 返回轨迹列表
    def _start(self, det):  # 新轨
        t=Track(self.next_id, det)  # 创建轨迹
        self.tracks[self.next_id]=t  # 注册
        self.next_id+=1  # 递增
    def _purge(self):  # 删除超时轨迹
        rm=[tid for tid,t in self.tracks.items() if t.lost>self.max_lost]  # 需删列表
        for tid in rm: del self.tracks[tid]  # 删除

# ============================ 多多边形 ROI 交互绘制（货架/区域） ============================
class MultiPolygonDrawer:  # 多 ROI 绘制
    def __init__(self, win_name):  # 构造
        self.win = win_name  # 窗口名
        self.polys = []  # 已完成的多边形列表
        self.names = []  # 多边形名称列表
        self.curr = []  # 正在绘制的多边形点
        self.editing = False  # 是否处于绘制状态
        cv2.setMouseCallback(self.win, self._mouse_cb)  # 绑定鼠标回调
    def _mouse_cb(self, event, x, y, flags, param):  # 鼠标事件
        if not self.editing and event == cv2.EVENT_LBUTTONDOWN:  # 新建多边形第一点
            self.curr = [(x,y)]; self.editing = True  # 启动编辑
        elif self.editing and event == cv2.EVENT_LBUTTONDOWN:  # 编辑中左键加点
            self.curr.append((x,y))  # 追加点
        elif self.editing and event == cv2.EVENT_RBUTTONDOWN and len(self.curr)>=3:  # 右键闭合
            self._finish_poly()  # 完成多边形
    def _finish_poly(self):  # 完成多边形
        self.polys.append(self.curr.copy())  # 存储
        self.names.append(f"{SHELF_NAME_PREFIX}-{len(self.polys)}")  # 自动命名
        self.curr = []; self.editing=False  # 清空并退出编辑
    def reset(self):  # 重置所有
        self.polys = []; self.names = []; self.curr = []; self.editing=False  # 归零
    def draw(self, frame):  # 在帧上画出多边形
        # 画已完成的多边形
        for idx, pts in enumerate(self.polys):  # 遍历
            color=(0,200,0)  # 绿色
            for i in range(len(pts)):  # 连线
                cv2.line(frame, pts[i], pts[(i+1)%len(pts)], color, 2)  # 边
            # 在多边形第一个点附近写名称
            put_text_cn(frame, self.names[idx], (pts[0][0]+5, pts[0][1]+5), font_path=FONT_PATH, font_size=22, color=(80,220,80))  # 名称
        # 画正在绘制的多边形
        if self.editing and len(self.curr)>0:  # 若在编辑
            for p in self.curr: cv2.circle(frame, p, 4, (0,255,255), -1)  # 顶点
            for i in range(1,len(self.curr)): cv2.line(frame, self.curr[i-1], self.curr[i], (0,255,255), 2)  # 边
    def contains_point(self, idx, pt):  # 点是否在第 idx 个多边形内
        if idx<0 or idx>=len(self.polys): return False  # 越界
        pts = np.array(self.polys[idx], dtype=np.int32)  # 转数组
        return cv2.pointPolygonTest(pts, (float(pt[0]), float(pt[1])), False) >= 0  # >=0 表示在内或边上
    def mask(self, shape):  # 生成整体布尔掩膜（所有 ROI）
        m = np.zeros(shape[:2], dtype=np.uint8)  # 零矩阵
        for pts in self.polys: cv2.fillPoly(m, [np.array(pts,np.int32)], 1)  # 填充为1
        return m.astype(bool)  # 布尔

# ============================ 零售指标管理（热力图 / 停留 / 到访 / 触达率） ============================
class RetailMetrics:  # 指标类
    def __init__(self, frame_shape, drawer: MultiPolygonDrawer):  # 构造
        h, w = frame_shape[:2]  # 高宽
        self.hd, self.wd = max(1,h//HEATMAP_DOWNSAMPLE), max(1,w//HEATMAP_DOWNSAMPLE)  # 下采样尺寸
        self.heatmap = np.zeros((self.hd, self.wd), dtype=np.float32)  # 热力图累积矩阵
        self.drawer = drawer  # ROI 管理
        self.all_visitors = set()  # 所有出现过的唯一 track_id
        self.roi_visitors = []  # 各 ROI 的唯一访客集合
        self.roi_visits = []  # 各 ROI 进入事件计数
        self.dwell = {}  # 停留时长 { (tid, roi_idx) : 累计秒 }
        self.inside_state = {}  # 状态 { (tid, roi_idx) : bool }
        self.enter_ts = {}  # 进入时间戳 { (tid, roi_idx) : t }
        self._sync_roi_structs()  # 初始化与 ROI 数一致的结构
    def _sync_roi_structs(self):  # 同步 ROI 相关数组大小
        n = len(self.drawer.polys)  # ROI 数
        self.roi_visitors = [set() for _ in range(n)]  # 每个 ROI 的唯一访客集合
        self.roi_visits  = [0 for _ in range(n)]  # 每个 ROI 的进入次数
    def update_heatmap(self, tracks):  # 更新热力图
        for t in tracks:  # 遍历轨迹
            cx, cy = t.centroid()  # 中心
            x = np.clip(cx//HEATMAP_DOWNSAMPLE, 0, self.wd-1)  # 下采样 x
            y = np.clip(cy//HEATMAP_DOWNSAMPLE, 0, self.hd-1)  # 下采样 y
            self.heatmap[y, x] += 1.0  # 计数+1
    def update_dwell_and_visits(self, tracks, dt, csv_writer):  # 更新停留/到访/触达
        for t in tracks:  # 遍历轨迹
            self.all_visitors.add(t.id)  # 记录唯一访客
            for ri in range(len(self.drawer.polys)):  # 遍历所有 ROI
                key = (t.id, ri)  # 键
                inside_now = self.drawer.contains_point(ri, t.centroid())  # 当前是否在 ROI 内
                inside_prev = self.inside_state.get(key, False)  # 上一状态
                if inside_now:  # 在 ROI 内
                    self.dwell[key] = self.dwell.get(key, 0.0) + dt  # 累加停留时长
                if (not inside_prev) and inside_now:  # 发生进入
                    self.roi_visits[ri] += 1  # 进入次数+1
                    self.roi_visitors[ri].add(t.id)  # 记录唯一访客
                    self.enter_ts[key] = time.time()  # 记录进入时间戳
                    # 写入进入事件
                    x1,y1,x2,y2 = map(int, t.bbox)  # 框
                    csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "enter", t.id,
                                         self.drawer.names[ri], x1,y1,x2,y2, f"{t.score:.4f}"])  # CSV
                if inside_prev and (not inside_now):  # 发生离开
                    # 写入离开事件与该次停留时长（秒）
                    dwell_sec = self.dwell.get(key, 0.0)  # 当前累计
                    csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "leave", t.id,
                                         self.drawer.names[ri], -1,-1,-1,-1, f"{dwell_sec:.2f}"])  # CSV
                    if key in self.enter_ts: del self.enter_ts[key]  # 清理进入时间戳
                self.inside_state[key] = inside_now  # 更新状态
    def draw_overlays(self, frame, show_heatmap=True, show_trails=True):  # 绘制叠加层
        if show_heatmap and self.heatmap.max()>0:  # 绘制热力图
            hm_norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # 归一化
            hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)  # 伪彩色
            hm_color = cv2.resize(hm_color, (frame.shape[1], frame.shape[0]))  # 放大至帧大小
            frame[:] = cv2.addWeighted(frame, 1.0, hm_color, HEATMAP_ALPHA, 0)  # 叠加
        if show_trails:  # 绘制轨迹
            # 轨迹绘制在外部循环中完成（见主循环），此处只保留接口占位
            pass  # 无需额外处理
        # 绘制 ROI 边与名称
        self.drawer.draw(frame)  # 画 ROI
        # 绘制每个 ROI 的统计小条
        y0 = 10  # 起始 y
        for i in range(len(self.drawer.polys)):  # 遍历 ROI
            name = self.drawer.names[i]  # 名称
            visitors = len(self.roi_visitors[i])  # 唯一访客数
            visits = self.roi_visits[i]  # 进入次数
            reach = (visitors/len(self.all_visitors)*100.0) if len(self.all_visitors)>0 else 0.0  # 触达率%
            txt = f"{name}｜访客:{visitors}｜进入:{visits}｜触达率:{reach:.1f}%"  # 文本
            put_text_cn(frame, txt, (10, y0), font_path=FONT_PATH, font_size=22, color=(0,255,180))  # 绘制
            y0 += 26  # 下一行
    def save_heatmap(self, path, frame_shape):  # 保存热力图图片
        if self.heatmap.max()==0: return  # 无数据不保存
        hm_norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # 归一
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)  # 伪彩
        hm_color = cv2.resize(hm_color, (frame_shape[1], frame_shape[0]))  # 放大
        cv2.imwrite(path, hm_color)  # 写文件
    def export_dwell_summary(self, path):  # 导出停留时长汇总
        # 汇总为：track_id, roi_name, dwell_seconds
        with open(path, "w", newline="", encoding="utf-8") as f:  # 打开文件
            w = csv.writer(f)  # writer
            w.writerow(["track_id","roi_name","dwell_seconds"])  # 表头
            for (tid,ri),sec in self.dwell.items():  # 遍历键值
                w.writerow([tid, self.drawer.names[ri], f"{sec:.2f}"])  # 写行

# ============================ 主流程 ============================
def main():  # 入口函数
    cap = cv2.VideoCapture(1)  # 打开默认摄像头
    if not cap.isOpened(): raise RuntimeError("无法打开摄像头 0，请检查设备/权限。")  # 打开失败报错

    # 先获取一帧确定分辨率（用于热力图尺寸等）
    ok, frame = cap.read()  # 读一帧
    if not ok: raise RuntimeError("读取摄像头首帧失败。")  # 首帧失败
    H, W = frame.shape[:2]  # 高宽

    # 初始化模型与跟踪器
    model = YOLOv8ONNX(ONNX_PATH, use_cuda=USE_CUDA)  # 创建推理器
    print("[INFO] ORT providers:", model.session.get_providers())  # 打印执行器
    tracker = IOUTracker(iou_thres=0.3, max_lost=20, min_area=15*15)  # 跟踪器

    # ROI 交互
    win = "零售洞察-绘制ROI：左键加点，右键闭合；回车完成；r重置；q退出"  # 窗口名
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)  # 可调窗口
    drawer = MultiPolygonDrawer(win)  # 多多边形绘制器

    # 提示阶段：先绘制 ROI
    while True:  # ROI 绘制循环
        ok, frame = cap.read()  # 读帧
        if not ok: continue  # 失败继续
        drawer.draw(frame)  # 画 ROI
        # 操作提示
        put_text_cn(frame, "操作: 左键加点  右键闭合  回车开始统计  r重置  q退出",
                    (10, 28), font_path=FONT_PATH, font_size=24, color=(0,255,255))  # 提示
        # 显示
        cv2.imshow(win, frame)  # 展示
        key = cv2.waitKey(1) & 0xFF  # 键
        if key == ord('r'): drawer.reset()  # 重置所有 ROI
        elif key == ord('q'):  # 退出
            cap.release(); cv2.destroyAllWindows(); return  # 直接结束
        elif key == 13:  # Enter 回车键
            if len(drawer.polys)>=1: break  # 至少一个 ROI 才进入
            else: put_text_cn(frame, "请至少绘制一个ROI后回车", (10, 60), font_path=FONT_PATH, font_size=24, color=(0,200,200))  # 提示

    # 初始化指标管理与 CSV
    metrics = RetailMetrics((H,W,3), drawer)  # 指标管理对象
    events_file_exists = Path(EVENTS_CSV).exists()  # 是否已有事件文件
    f_events = open(EVENTS_CSV, "a", newline="", encoding="utf-8")  # 打开事件 CSV
    w_events = csv.writer(f_events)  # writer
    if not events_file_exists: w_events.writerow(["timestamp","event","track_id","roi_name","x1","y1","x2","y2","value"])  # 表头

    # 运行阶段窗口
    run_win = "零售洞察-运行：h热力图开关  t轨迹开关  s存热力图  q退出"  # 窗口名
    cv2.namedWindow(run_win, cv2.WINDOW_NORMAL)  # 创建窗口
    show_heatmap, show_trails = True, True  # 开关
    fps = 0.0  # FPS
    last_t = time.time()  # 上一帧时间
    color_map = {}  # 轨迹颜色表

    # 主循环：检测-跟踪-指标
    while True:  # 主循环
        ok, frame = cap.read()  # 读取一帧
        if not ok:  # 若失败
            print("[WARN] 读取摄像头失败，继续...")  # 警告
            continue  # 下一帧

        t0 = time.time()  # 当前时间
        dt = t0 - last_t  # 帧间隔
        last_t = t0  # 更新时间基准

        dets = model.infer(frame)  # 推理得到 [N,6]
        tracks = tracker.update(dets)  # 跟踪更新

        # 更新热力图 / 停留时长 / 到访 / 触达
        metrics.update_heatmap(tracks)  # 热力图累积
        metrics.update_dwell_and_visits(tracks, dt, w_events)  # 停留/事件

        # 绘制目标框/轨迹
        for t in tracks:  # 遍历轨迹
            if t.id not in color_map:  # 分配颜色
                color_map[t.id] = tuple(int(c) for c in np.random.randint(0,255,size=3))  # 随机
            color = color_map[t.id]  # 取色
            x1,y1,x2,y2 = map(int, t.bbox)  # 框
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)  # 画框
            cx,cy = t.centroid()  # 中心
            cv2.circle(frame, (cx,cy), 3, color, -1)  # 中心点
            put_text_cn(frame, f"ID {t.id} 行人 {t.score:.2f}", (x1, max(0,y1-24)), font_path=FONT_PATH, font_size=22, color=color)  # 标签
            if show_trails and len(t.trail)>=2:  # 画轨迹
                for i in range(1, len(t.trail)):  # 连线
                    cv2.line(frame, t.trail[i-1], t.trail[i], color, 2)  # 轨迹线

        # 绘制热力图 + ROI统计条
        metrics.draw_overlays(frame, show_heatmap=show_heatmap, show_trails=show_trails)  # 叠加层

        # 左上角状态/帧率
        fps = 0.9*fps + 0.1*(1.0/max(1e-6, time.time()-t0))  # 平滑 FPS
        put_text_cn(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-32), font_path=FONT_PATH, font_size=24, color=(0,200,255))  # 帧率展示

        # 显示
        cv2.imshow(run_win, frame)  # 展示结果
        key = cv2.waitKey(1) & 0xFF  # 按键
        if key == ord('h'): show_heatmap = not show_heatmap  # 热力图开关
        elif key == ord('t'): show_trails = not show_trails  # 轨迹开关
        elif key == ord('s'):  # 保存热力图
            metrics.save_heatmap(HEATMAP_SNAPSHOT, frame.shape)  # 写图
            print(f"[INFO] 已保存热力图到 {HEATMAP_SNAPSHOT}")  # 控制台提示
        elif key == ord('q'):  # 退出
            break  # 跳出循环

    # 导出停留时长汇总 & 资源清理
    metrics.export_dwell_summary(DWELL_SUMMARY_CSV)  # 导出停留汇总 CSV
    f_events.close()  # 关事件 CSV
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关窗

# ============================ 程序入口 ============================
if __name__ == "__main__":  # 入口判断
    if not Path(ONNX_PATH).exists():  # 检查模型路径
        raise FileNotFoundError(f"未找到 ONNX 模型：{ONNX_PATH}")  # 提示
    # FONT_PATH = "C:/Windows/Fonts/simhei.ttf"  # 建议显式指定中文字体（示例）
    if FONT_PATH is None and _auto_pick_font() is None:  # 字体检测
        print("提示：未设置 FONT_PATH，且未能自动找到中文字体；请设置为本机字体路径，以避免中文无法显示。")  # 控制台提示
    main()  # 启动主流程
