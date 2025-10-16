# -*- coding: utf-8 -*-  # 指定源码文件使用 UTF-8 编码，确保中文注释与字符串正常显示
# ============================================================
# YOLOv8 (ONNX) 摄像头实时推理
# 依赖安装：pip install onnxruntime opencv-python numpy
# 功能概要：加载 YOLOv8 的 ONNX 模型，打开摄像头，进行实时检测并可视化结果
# 兼容两类常见导出：A) 已包含 NMS 的 ONNX 输出；B) 原始预测张量（需在本地做解码+NMS）
# ============================================================

import time  # 引入 time 模块，用于计算每帧耗时与估算 FPS
import cv2  # 引入 OpenCV 库，用于图像读取、摄像头采集与绘制可视化
import numpy as np  # 引入 NumPy，用于高效的张量/数组运算
import onnxruntime as ort  # 引入 onnxruntime，用于加载和推理 ONNX 模型
from pathlib import Path  # 引入 Path 类，便于跨平台处理文件路径

# ============================ 配置区域 ============================
ONNX_PATH = "yolov8n.onnx"  # 指定 ONNX 模型文件路径（请替换为你的 .onnx 文件路径）
IMG_SIZE = (640, 640)  # 模型的输入分辨率，(width, height)，需要与导出时保持一致
CONF_THRES = 0.25  # 置信度阈值：原始输出解码时，低于该阈值的候选框会被丢弃
IOU_THRES = 0.45  # NMS 的 IoU 阈值：重叠超过该阈值的候选框将被抑制
CLASS_NAMES = [  # COCO 数据集的 80 类类别名列表（若是自定义数据集，请替换为你的类别名列表）
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
]  # 以上为 COCO 的标准 80 类名；若模型类别数不同，请务必同步修改

# ============================ 工具函数 ============================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """将图像按比例缩放并使用指定颜色填充到目标尺寸，返回处理后图像、缩放比例与填充值。"""  # 函数注释：描述功能
    shape = im.shape[:2]  # 取原图的高宽 (h, w)
    if isinstance(new_shape, int):  # 如果传入的是单个整数
        new_shape = (new_shape, new_shape)  # 则将其扩展为方形 (w, h)
    new_w, new_h = new_shape  # 解包目标尺寸 (width, height)

    r = min(new_w / shape[1], new_h / shape[0])  # 计算缩放比例 r，确保等比缩放后不超过目标尺寸
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # 计算缩放后但未填充的尺寸 (w, h)
    dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]  # 计算需要填充的宽高差值
    dw /= 2  # 左右平均填充
    dh /= 2  # 上下平均填充

    if shape[::-1] != new_unpad:  # 如果缩放后的尺寸与目标尺寸不同
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)  # 对图像进行缩放插值
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上、下边需要填充的像素数
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左、右边需要填充的像素数
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 执行常数值（灰色）边框填充
    return im, r, (left, top)  # 返回处理后图像、缩放比例 r、左上角填充偏移（用于反变换坐标）

def nms_boxes(boxes, scores, iou_thres=0.5):
    """纯 NumPy 实现的 NMS；输入为候选框与其分数，返回保留下来的索引列表。"""  # 函数注释
    x1, y1, x2, y2 = boxes.T  # 将 boxes 的列拆分为四个向量：左上与右下角坐标
    areas = (x2 - x1) * (y2 - y1)  # 计算每个候选框的面积
    order = scores.argsort()[::-1]  # 将分数从大到小排序，得到索引顺序

    keep = []  # 存储最终保留的候选框索引
    while order.size > 0:  # 只要还有未处理的候选框
        i = order[0]  # 取当前分数最高的索引
        keep.append(i)  # 将其加入保留列表
        if order.size == 1:  # 若已处理完毕（只剩一个）
            break  # 跳出循环
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 计算与其余框交集的左上角 x
        yy1 = np.maximum(y1[i], y1[order[1:]])  # 计算与其余框交集的左上角 y
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 计算与其余框交集的右下角 x
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 计算与其余框交集的右下角 y

        w = np.maximum(0.0, xx2 - xx1)  # 交集宽度，负数需截断为 0
        h = np.maximum(0.0, yy2 - yy1)  # 交集高度，负数需截断为 0
        inter = w * h  # 交集面积
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)  # 计算 IoU，分母加 1e-6 防止除零

        inds = np.where(iou <= iou_thres)[0]  # 找到 IoU 小于阈值的索引（这些保留继续参与）
        order = order[inds + 1]  # 更新排序队列（跳过当前最大分数的索引）
    return keep  # 返回保留索引列表

def scale_coords(boxes, ratio, pad):
    """将基于 letterbox 尺度的坐标反变换回原图尺度。"""  # 函数注释
    boxes[:, [0, 2]] -= pad[0]  # 将 x1/x2 减去左侧填充
    boxes[:, [1, 3]] -= pad[1]  # 将 y1/y2 减去上侧填充
    boxes[:, :4] /= ratio  # 再除以缩放比例 r，映射回原图尺寸
    return boxes  # 返回变换后的坐标

def xywh2xyxy(xywh):
    """将 (cx, cy, w, h) 中心点与宽高格式转换为 (x1, y1, x2, y2) 左上与右下角格式。"""  # 函数注释
    xyxy = np.zeros_like(xywh)  # 创建与输入同形状的数组用于存放结果
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # 计算 x1 = cx - w/2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # 计算 y1 = cy - h/2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # 计算 x2 = cx + w/2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # 计算 y2 = cy + h/2
    return xyxy  # 返回转换后的坐标

# ============================ 模型封装类 ============================
class YOLOv8ONNX:
    """封装 ONNXRuntime 推理的 YOLOv8 模型，统一输出格式为 [N, 6](x1,y1,x2,y2,score,cls)。"""  # 类注释

    def __init__(self, onnx_path, use_cuda=True):
        """构造函数：创建 ORT 推理会话，记录输入输出信息。"""  # 方法注释
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]  # 优先使用 GPU 提供者，失败则回退 CPU
        try:  # 尝试创建带指定 providers 的推理会话
            self.session = ort.InferenceSession(Path(onnx_path).as_posix(), providers=providers)  # 创建 ORT 会话
        except Exception:  # 如果创建失败（例如本机无 CUDA 版 ORT）
            self.session = ort.InferenceSession(Path(onnx_path).as_posix(), providers=["CPUExecutionProvider"])  # 回退到纯 CPU 推理
        self.input_name = self.session.get_inputs()[0].name  # 取第一个输入张量的名称（ONNXRuntime 前向需要字典键为输入名）
        self.outputs_info = [(o.name, o.shape) for o in self.session.get_outputs()]  # 记录所有输出的名字与形状（用于判断导出类型）

    def infer(self, img_bgr):
        """对单帧 BGR 图像进行推理，返回原图尺度的检测结果数组 [N, 6]。"""  # 方法注释
        img0 = img_bgr  # 将传入图像命名为 img0，以便后续保持清晰
        lb_img, ratio, pad = letterbox(img0, (IMG_SIZE[0], IMG_SIZE[1]))  # 执行 letterbox 预处理，得到缩放后图像、缩放比例和填充偏移
        img_rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)  # 将 BGR 转为 RGB，因为大多数模型以 RGB 训练
        img = img_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # HWC->CHW，再添加 batch 维度，转 float32 并归一化到 [0,1]

        outputs = self.session.run(None, {self.input_name: img})  # 执行前向推理，None 表示取所有输出；输入是 dict：{输入名: 输入张量}

        dets = None  # 初始化检测结果变量，用于最终统一为 [N,6]
        if len(outputs) == 1:  # 若只有一个输出张量
            out = outputs[0]  # 取该输出
            if out.ndim == 2 and out.shape[1] in (6, 7):  # 情况 A：形状为 [N,6] 或 [N,7]（有些导出在第 0 列放 batch_id）
                if out.shape[1] == 7:  # 若包含 batch_id
                    out = out[:, 1:]  # 去掉 batch_id 列，保留 [x1,y1,x2,y2,score,cls]
                dets = out  # 将结果直接作为检测结果
            elif out.ndim == 3 and out.shape[-1] in (6, 7):  # 情况 A 变体：形状为 [1,N,6/7]
                out = out[0]  # 去掉 batch 维度
                if out.shape[1] == 7:  # 同样处理可能存在的 batch_id 列
                    out = out[:, 1:]
                dets = out  # 赋值检测结果
            elif out.ndim == 3 and out.shape[0] == 1 and out.shape[1] >= 5:  # 情况 B：原始输出，常见为 [1,84,8400]
                dets = self._postprocess_raw(out[0], ratio, pad)  # 对原始输出做本地解码（置信度过滤 + NMS），得到 letterbox 尺度的 dets
        else:  # 若存在多个输出张量（不同导出版本可能会多输出）
            det = None  # 临时变量，用于尝试找到已含 NMS 的输出张量
            for o in outputs:  # 遍历所有输出
                if o.ndim == 2 and o.shape[1] in (6, 7):  # 优先匹配已含 NMS 的 2D 输出
                    det = o if o.shape[1] == 6 else o[:, 1:]  # 若是 7 列则去掉首列 batch_id
                    break  # 找到了即可退出
                if o.ndim == 3 and o.shape[-1] in (6, 7):  # 或者匹配 [1,N,6/7] 的 3D 输出
                    det = o[0] if o.shape[-1] == 6 else o[0][:, 1:]  # 同理处理 batch 维与 batch_id
                    break  # 找到即退出
            if det is not None:  # 如果找到了已含 NMS 的输出
                dets = det  # 直接使用
            else:  # 否则尝试原始输出的解码路径
                for o in outputs:  # 再遍历一次以匹配原始输出形状
                    if o.ndim == 3 and o.shape[0] == 1 and o.shape[1] >= 5:  # 常见为 [1,84,8400]
                        dets = self._postprocess_raw(o[0], ratio, pad)  # 做解码与 NMS
                        break  # 处理后退出循环

        if dets is None or dets.size == 0:  # 若没有检测到任何结果
            return np.zeros((0, 6), dtype=np.float32)  # 返回空的 [0,6] 数组，统一接口

        boxes = dets[:, :4].copy()  # 拷贝出候选框坐标（避免原数组被就地修改）
        scores = dets[:, 4]  # 取出置信度分数列
        cls_id = dets[:, 5]  # 取出类别 id 列
        boxes = scale_coords(boxes, ratio, pad)  # 将坐标从 letterbox 尺度映射回原图尺度

        dets_final = np.concatenate([boxes, scores[:, None], cls_id[:, None]], axis=1)  # 重新拼接为 [N,6] 的最终结果
        return dets_final  # 返回在原图尺度上的检测结果

    def _postprocess_raw(self, raw, ratio, pad):
        """对原始输出做后处理：选择最佳类别、置信度阈值过滤并按类别执行 NMS，输出 [N,6]（在 letterbox 尺度）。"""  # 方法注释
        pred = raw.transpose(1, 0)  # 将 (C, Num) 转置为 (Num, C)，便于逐候选框处理，典型形状从 [84,8400] 变为 [8400,84]
        boxes_xywh = pred[:, :4]  # 取出每个候选框的 (cx, cy, w, h)
        cls_scores = pred[:, 4:]  # 取出各类别的分数矩阵，形状 [Num, nc]
        if cls_scores.size == 0:  # 若没有类别分数（异常情况）
            return np.zeros((0, 6), dtype=np.float32)  # 返回空结果

        cls_id = np.argmax(cls_scores, axis=1)  # 对每个候选框，选择分数最高的类别索引
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_id]  # 提取对应的最大类别分数作为该框的置信度

        keep = scores >= CONF_THRES  # 基于置信度阈值进行一次初筛
        if not np.any(keep):  # 若没有任何候选框通过阈值
            return np.zeros((0, 6), dtype=np.float32)  # 返回空结果

        boxes_xywh = boxes_xywh[keep]  # 仅保留通过置信度阈值的候选框
        scores = scores[keep]  # 同步保留对应的置信度
        cls_id = cls_id[keep]  # 同步保留对应的类别 id

        boxes_xyxy = xywh2xyxy(boxes_xywh)  # 将 (cx,cy,w,h) 转为 (x1,y1,x2,y2)，便于 NMS 处理

        final_boxes = []  # 用于累积各类别经 NMS 后的框
        final_scores = []  # 用于累积各类别经 NMS 后的置信度
        final_cls = []  # 用于累积各类别 id

        for c in np.unique(cls_id):  # 遍历每个出现过的类别
            idxs = np.where(cls_id == c)[0]  # 找到该类别对应的索引集合
            b = boxes_xyxy[idxs]  # 取出该类别的候选框
            s = scores[idxs]  # 取出该类别的分数
            keep_idx = nms_boxes(b, s, IOU_THRES)  # 对该类别的候选框执行 NMS，得到保留索引
            final_boxes.append(b[keep_idx])  # 叠加保留的框
            final_scores.append(s[keep_idx])  # 叠加保留的分数
            final_cls.append(np.full((len(keep_idx),), c, dtype=np.float32))  # 叠加对应类别 id，长度与保留框一致

        if len(final_boxes) == 0:  # 若所有类别都没有保留框
            return np.zeros((0, 6), dtype=np.float32)  # 返回空结果

        boxes = np.concatenate(final_boxes, axis=0)  # 将各类别的框在第 0 维拼接
        scores = np.concatenate(final_scores, axis=0)  # 将各类别的置信度拼接
        cls = np.concatenate(final_cls, axis=0)  # 将各类别 id 拼接
        return np.concatenate([boxes, scores[:, None], cls[:, None]], axis=1).astype(np.float32)  # 拼接为 [N,6] 并返回

# ============================ 主流程：摄像头循环 ============================
def main():
    """主函数：打开摄像头，循环读取帧并做推理与可视化。"""  # 函数注释
    cap = cv2.VideoCapture(0)  # 打开默认摄像头（索引 0）；若有多个摄像头可改为 1/2/...
    if not cap.isOpened():  # 检查摄像头是否成功打开
        raise RuntimeError("无法打开摄像头 0，请检查设备是否存在或权限是否被允许。")  # 抛出异常提示用户检查设备/权限

    model = YOLOv8ONNX(ONNX_PATH, use_cuda=True)  # 创建模型实例，默认尝试使用 GPU 推理（若不可用会自动回退 CPU）
    print("[INFO] ONNX providers:", model.session.get_providers())  # 打印当前 ORT 实际使用的 providers（便于确认是否走了 GPU）

    fps = 0.0  # 初始化 FPS 值（会做指数滑动平均以更平滑显示）

    while True:  # 进入主循环，不断读取摄像头帧进行推理
        ok, frame = cap.read()  # 从摄像头读取一帧图像
        if not ok:  # 如果读取失败
            print("[WARN] 读取摄像头失败，继续尝试下一帧...")  # 打印警告信息
            continue  # 跳过本次循环

        t0 = time.time()  # 记录当前时间，用于计算单帧推理耗时
        dets = model.infer(frame)  # 调用模型进行推理，得到在原图尺度的检测结果 [N,6]
        t1 = time.time()  # 记录推理结束时间

        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, t1 - t0))  # 使用指数滑动平均计算 FPS，避免抖动并防止除零

        for *xyxy, conf, cls_id in dets:  # 遍历每个检测到的目标，解包出坐标、置信度与类别
            x1, y1, x2, y2 = map(int, xyxy)  # 将浮点坐标转换为整数，便于绘制矩形框
            cls_id = int(cls_id)  # 将类别 id 转为整数类型
            # 生成标签文本：类别名称 + 置信度（保留两位小数）；若类别 id 越界则直接显示 id 数值
            label = f"{CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else cls_id}:{conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 在原图上绘制绿色矩形框（线宽 2）
            cv2.putText(frame, label, (x1, max(0, y1 - 5)),  # 在框上方绘制文本标签，纵坐标做下限裁剪避免越界
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)  # 指定字体、字号、颜色、线宽与抗锯齿

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),  # 在左上角显示当前帧率（保留 1 位小数）
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)  # 使用黄色文本便于识别

        cv2.imshow("YOLOv8-ONNX Webcam", frame)  # 显示结果窗口，窗口名为 "YOLOv8-ONNX Webcam"
        key = cv2.waitKey(1) & 0xFF  # 等待 1ms 并获取键盘键值（与 0xFF 按位与避免高位影响）
        if key == ord('q'):  # 若按下 'q' 键
            break  # 跳出循环，准备收尾

    cap.release()  # 释放摄像头资源
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 创建的窗口

# ============================ 脚本入口 ============================
if __name__ == "__main__":  # 判断当前文件是否作为主程序运行
    if not Path(ONNX_PATH).exists():  # 在运行前检查模型文件是否存在
        raise FileNotFoundError(f"未找到 ONNX 模型文件: {ONNX_PATH}，请将路径改为你的 .onnx 文件。")  # 未找到则抛错提醒
    main()  # 若文件存在，调用主函数，开始摄像头推理流程
