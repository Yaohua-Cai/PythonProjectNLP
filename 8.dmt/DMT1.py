# 文本（Text）
# 用任意 tokenizer（如 BPE/WordPiece）得到 token_ids: LongTensor[B, L]；若长度不一需 pad 并构造 attention_mask（上面示例为简化，若要严格 mask 可改用 nn.TransformerEncoder 的 src_key_padding_mask）。
#
# 图像（Image）
# 将图片读为 float32 张量 [3,H,W]，归一化（如 mean/std）；分辨率不一可统一 resize（与 patch 配合）。
#
# 语音（Audio）
# 直接用波形 [1,T] 即可（示例用 1D patch 卷积），或先用你熟悉的特征（Log-Mel、MFCC 等）替换为 [C,T] 并把 PatchEmbed1D 的 in_ch 改为 C。
#
# 视频（Video）
# 抽帧得到 [Tm,3,H,W]（例如每秒采 2~4 帧）；如需更强时序建模，替换 VideoEncoder 为 TimeSformer/ViViT 等。
#
# 跨模态交互
# 当前对全局向量（每模态的 [CLS]）做双向交互，轻量稳定。你也可以：
#
# 用序列级交互（让一模态序列作为 q，另一模态序列作 k/v），效果更强、显存更大；
# 叠加多轮 Mixer（把 layers 增大），或仅对关键模态对（如 T-I、T-A）做交互。
#
# 对比学习配置
# 现在把 T-I/T-A/T-V/I-V/I-A/A-V 6 对等权平均；可按业务加权或只保留你关心的配对；
# logit_scale = exp(learnable) 自动学习温度，初始化为 1.0 可把 logit_scale.data.fill_(math.log(1/0.07)) 做成 CLIP 风格。
#
# 多卡与混合精度
# 已启用 AMP（CUDA 自动混合精度）；多卡可用 DDP，你只需把 DataLoader 和 model 包装进 DDP 训练模板即可。


# -*- coding: utf-8 -*-
"""
多模态对比学习（Text / Image / Audio / Video）+ 基于 Transformer 的跨模态注意力

功能概览：
1. 四模态编码器：TextEncoder / ImageEncoder / AudioEncoder / VideoEncoder
2. 跨模态交互：CrossModalMixer（使用多轮多头注意力，对各模态全局向量做深度交互）
3. 训练目标：多对 InfoNCE 对比学习（T-I, T-A, T-V, I-V, I-A, A-V）
4. 训练过程：AMP 混合精度（CUDA 自动启用），AdamW 优化器
5. 全流程：定义 → 训练 → quick_eval → 保存 → 加载 → 推理 sanity check

你可以：
- 替换合成数据为真实数据（图像/音频/视频路径与文本 token）
- 替换简化编码器为预训练骨干（BERT/ViT/WavLM/TimeSformer 等）
"""

# ====== 标准库 / 类型提示 ======
import math                   # 数学函数（如对数/指数、pi 等）
import random                 # 随机数种子与简单随机操作
from typing import Tuple, Dict  # 类型注解（提升代码可读性）

# ====== PyTorch 基础 ======
import torch                  # PyTorch 主库
import torch.nn as nn         # 神经网络模块（层、容器、损失等）
import torch.nn.functional as F  # 函数式 API（激活、损失、归一化等）


# =============================================================================
# 基础工具模块
# =============================================================================

def get_device():
    """
    返回当前可用的设备：
    - 若存在 CUDA GPU：返回 torch.device("cuda")
    - 否则返回 CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动判断 GPU/CPU

class PositionalEncoding(nn.Module):
    """
    正弦/余弦 位置编码（Sinusoidal Positional Encoding）

    作用：
    - 为 Transformer 这类“无卷积/无递归”的结构引入位置信息
    - 可泛化到任意长度序列（通过公式生成，不依赖学习参数）

    参数：
    - d_model: 每个 token/patch 的通道维度（模型维度）
    - max_len: 预生成的最大序列长度上限（足够大即可）

    前向：
    - 输入形状 [B, L, D]，输出同形状，在通道维上叠加位置编码
    """
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()                                          # 调用父类初始化
        pe = torch.zeros(max_len, d_model)                          # 创建 [L, D] 零张量存放位置编码
        position = torch.arange(0, max_len, dtype=torch.float32)    # 生成位置索引 [0,1,2,...,L-1]
        position = position.unsqueeze(1)                            # 扩展为 [L, 1] 便于广播
        div_term = torch.exp(                                       # 计算频率项（指数递减）
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / max_len)
        )
        pe[:, 0::2] = torch.sin(position * div_term)                # 偶数维用正弦
        pe[:, 1::2] = torch.cos(position * div_term)                # 奇数维用余弦
        self.register_buffer("pe", pe.unsqueeze(0))                 # 注册为缓冲区 [1, L, D]

    def forward(self, x: torch.Tensor):
        # x: [B, L, D] —— 批、长度、通道
        L = x.size(1)                                               # 当前序列长度
        return x + self.pe[:, :L, :]                                # 叠加对应长度的位置编码后返回


class CLSHead(nn.Module):
    """
    序列汇聚策略（两种模式之一）：
    1) 使用可学习 [CLS] token：拼到序列最前，后续直接取该位置向量作为全局表征
    2) 使用平均池化：先对输入序列做均值池化得到一个全局 token，再与原序列拼接

    参数：
    - d_model: 模型维度
    - use_cls_token: 是否启用可学习 [CLS]（True）或采用均值池化（False）

    前向：
    - 输入 [B, L, D]，输出 [B, (1+L), D]（前面多 1 个全局 token）
    """
    def __init__(self, d_model: int, use_cls_token: bool = True):
        super().__init__()                                          # 初始化父类
        self.use_cls = use_cls_token                                # 记录模式
        if self.use_cls:                                            # 若使用可学习 CLS
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # 声明可学习参数 [1,1,D]
            nn.init.trunc_normal_(self.cls_token, std=0.02)         # 截断正态初始化
        else:
            self.cls_token = None                                   # 否则不创建该参数

    def forward(self, x: torch.Tensor):
        # x: [B, L, D]
        if self.use_cls:                                            # 使用可学习 CLS
            B = x.size(0)                                           # 批大小
            cls = self.cls_token.expand(B, -1, -1)                  # 扩展到 [B,1,D]
            return torch.cat([cls, x], dim=1)                       # 拼接为 [B,1+L,D]
        else:
            pooled = x.mean(dim=1, keepdim=True)                    # 均值池化得到 [B,1,D]
            return torch.cat([pooled, x], dim=1)                    # 拼接为 [B,1+L,D]


# =============================================================================
# 文本编码器：Embedding + Position + TransformerEncoder + 取 [CLS]
# =============================================================================

class TextEncoder(nn.Module):
    """
    文本编码器（简化示例）：

    结构：
    - 词嵌入层（Embedding）
    - 位置编码（PositionalEncoding）
    - TransformerEncoder（多层 Self-Attention）
    - CLSHead 注入全局 token
    - 输出：取全局 token（序列第 1 位）的向量作为文本全局表征

    参数：
    - vocab_size: 词表大小（演示用）
    - d_model: 模型通道维
    - nhead: 多头注意力头数
    - depth: 编码层堆叠深度
    - max_len: 最大序列长度（决定位置编码上限）
    """
    def __init__(self, vocab_size=32000, d_model=256, nhead=8, depth=4, max_len=512):
        super().__init__()                                          # 父类初始化
        self.token_emb = nn.Embedding(vocab_size, d_model)          # 词嵌入：将 token id 投到 d_model 维
        self.pos = PositionalEncoding(d_model, max_len=max_len+1)   # 位置编码（+1 为预留 CLS）
        encoder_layer = nn.TransformerEncoderLayer(                 # 定义单层 Transformer 编码层
            d_model=d_model, nhead=nhead,
            dim_feedforward=4*d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)  # 堆叠多层
        self.cls_head = CLSHead(d_model, use_cls_token=True)        # 启用可学习 CLS

    def forward(self, token_ids: torch.Tensor):
        """
        输入：
        - token_ids: [B, L] 的长整型张量（每个元素是词 ID）

        输出：
        - [B, D] 的全局文本向量（取 CLS 位置）
        """
        x = self.token_emb(token_ids)            # [B,L,D] 词嵌入
        x = self.cls_head(x)                     # [B,1+L,D] 注入 CLS
        x = self.pos(x)                          # [B,1+L,D] 加位置编码
        x = self.encoder(x)                      # [B,1+L,D] 通过 TransformerEncoder
        return x[:, 0, :]                        # [B,D] 取 CLS 位置作为全局文本表征


# =============================================================================
# 图像编码器：Conv2d Patch → 序列 → 位置编码 → TransformerEncoder → CLS
# =============================================================================

class PatchEmbed2D(nn.Module):
    """
    图像 Patch 嵌入（Vision Transformer 风格的简化版）：

    作用：
    - 使用 Conv2d(kernel=stride=patch) 将图像切成不重叠 patch，并投影到 d_model 维
    - 输出 token 序列（每个 patch 对应一个 token）

    输入：
    - 图像张量 [B, C, H, W]

    输出：
    - patch 序列 [B, N, D]，其中 N = (H/patch)*(W/patch)
    """

    def __init__(self, in_ch=3, patch=16, d_model=256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)                      # [B,D,h,w]
        # 原来：x = x.flatten(2).transpose(1, 2)
        # 修复：确保转置后是连续的
        x = x.flatten(2).transpose(1, 2).contiguous()   # [B,N,D] 现在是连续内存
        return x


class ImageEncoder(nn.Module):
    """
    图像编码器（简化示例）：

    结构：
    - PatchEmbed2D：把图像切成 patch，并映射到 token 序列
    - 位置编码 + TransformerEncoder
    - CLSHead 注入全局 token
    - 输出：CLS 位置向量作为图像全局表征
    """
    def __init__(self, d_model=256, nhead=8, depth=4, patch=16, in_ch=3, max_patches=1024):
        super().__init__()                                          # 父类初始化
        self.patch = PatchEmbed2D(in_ch=in_ch, patch=patch, d_model=d_model)  # Patch 嵌入
        self.pos = PositionalEncoding(d_model, max_len=max_patches+1)         # 位置编码
        encoder_layer = nn.TransformerEncoderLayer(                 # 单层 Transformer 编码层
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)  # 多层堆叠
        self.cls_head = CLSHead(d_model, use_cls_token=True)        # 可学习 CLS

    def forward(self, img: torch.Tensor):
        """
        输入：
        - img: [B, 3, H, W] 归一化图像

        输出：
        - [B, D] 图像全局表征
        """
        x = self.patch(img)                     # [B,N,D] patch 序列
        x = self.cls_head(x)                    # [B,1+N,D] 注入 CLS
        x = self.pos(x)                         # [B,1+N,D] 加位置编码
        x = self.encoder(x)                     # [B,1+N,D] Transformer 编码
        return x[:, 0, :]                       # [B,D] 取 CLS


# =============================================================================
# 音频编码器：Conv1d Patch → 序列 → 位置编码 → TransformerEncoder → CLS
# =============================================================================

class PatchEmbed1D(nn.Module):
    """
    1D 序列 Patch 嵌入（适用于原始波形或 1D 特征序列）：

    作用：
    - 使用 Conv1d(kernel=stride=patch) 将时间轴切块
    - 将每个块映射到 d_model 维，形成 token 序列

    输入：
    - [B, C, T]（C=通道，原始波形常用 C=1）

    输出：
    - [B, N, D] 序列，N = T // patch
    """
    def __init__(self, in_ch=1, patch=320, d_model=256):
        super().__init__()                                          # 父类初始化
        self.proj = nn.Conv1d(in_ch, d_model, kernel_size=patch, stride=patch)  # 时间轴切块

    def forward(self, x: torch.Tensor):
        # x: [B,C,T]
        x = self.proj(x)                         # [B,D,T/patch] 通过 1D 卷积切块
        x = x.transpose(1, 2)                    # -> [B,N,D] 与其他编码器对齐
        return x                                  # 返回序列


class AudioEncoder(nn.Module):
    """
    音频编码器（简化示例）：

    结构：
    - PatchEmbed1D：对 1D 序列（如原始波形、Log-Mel）切块成 token 序列
    - 位置编码 + TransformerEncoder
    - CLSHead 注入全局 token
    - 输出：CLS 位置向量作为音频全局表征

    备注：
    - 若使用多通道（如 80 维 Log-Mel），将 PatchEmbed1D 的 in_ch 改为对应通道数
    """
    def __init__(self, d_model=256, nhead=8, depth=4, patch=320, in_ch=1, max_patches=4000):
        super().__init__()                                          # 父类初始化
        self.patch = PatchEmbed1D(in_ch=in_ch, patch=patch, d_model=d_model)    # 1D 切块
        self.pos = PositionalEncoding(d_model, max_len=max_patches+1)          # 位置编码
        encoder_layer = nn.TransformerEncoderLayer(                 # 单层 Transformer
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)   # 堆叠多层
        self.cls_head = CLSHead(d_model, use_cls_token=True)        # 可学习 CLS

    def forward(self, wav: torch.Tensor):
        """
        输入：
        - wav: [B, 1, T] 原始波形（或其它 1D 特征，注意同步修改 in_ch）

        输出：
        - [B, D] 音频全局表征
        """
        x = self.patch(wav)                    # [B,N,D] token 序列
        x = self.cls_head(x)                   # [B,1+N,D] 注入 CLS
        x = self.pos(x)                        # [B,1+N,D] 位置编码
        x = self.encoder(x)                    # [B,1+N,D] Transformer 编码
        return x[:, 0, :]                      # [B,D] 取 CLS


# =============================================================================
# 视频编码器：逐帧图像 Patch → 展平为长序列 → 位置编码 → Transformer → CLS
# =============================================================================

class VideoEncoder(nn.Module):
    """
    视频编码器（简化示例）：

    结构：
    - 对每帧使用图像 PatchEmbed2D 得到若干 patch token
    - 将所有帧的 patch token 按时间拼接成一条更长的序列（无需 3D 卷积）
    - 位置编码 + TransformerEncoder
    - CLSHead 注入全局 token
    - 输出：CLS 位置向量作为视频全局表征

    备注：
    - 该结构轻量易懂；你也可替换为 TimeSformer/VideoMAE 等更强视频模型
    """
    def __init__(self, d_model=256, nhead=8, depth=4, patch=16, in_ch=3, max_tokens=4096):
        super().__init__()                                          # 父类初始化
        self.frame_patch = PatchEmbed2D(in_ch=in_ch, patch=patch, d_model=d_model)  # 帧级 patch
        self.pos = PositionalEncoding(d_model, max_len=max_tokens+1)               # 位置编码
        encoder_layer = nn.TransformerEncoderLayer(                 # 单层 Transformer
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)      # 多层堆叠
        self.cls_head = CLSHead(d_model, use_cls_token=True)        # 可学习 CLS

    def forward(self, vid: torch.Tensor):
        """
        输入：
        - vid: [B, T, 3, H, W] 视频张量（T 为帧数）

        输出：
        - [B, D] 视频全局表征
        """
        B, T, C, H, W = vid.shape          # 解包形状参数
        vid = vid.reshape(B*T, C, H, W)    # 合并时间维，逐帧处理 -> [B*T,3,H,W]
        tokens = self.frame_patch(vid)     # 帧内 patch 序列 -> [B*T, Np, D]
        Np = tokens.size(1)                # 每帧 patch 数
        tokens = tokens.view(B, T*Np, -1)  # 还原批次，按时间拼接 -> [B, T*Np, D]
        x = self.cls_head(tokens)          # 注入 CLS -> [B,1+T*Np,D]
        x = self.pos(x)                    # 位置编码
        x = self.encoder(x)                # Transformer 编码
        return x[:, 0, :]                  # 取 CLS -> [B,D]


# =============================================================================
# 跨模态注意力模块：对全局向量进行多轮跨模态深度交互
# =============================================================================

class CrossModalBlock(nn.Module):
    """
    跨模态交互基本单元（简化）：

    思路：
    - 使用 MultiheadAttention，让“查询模态”的全局向量 q，与“被融合模态”的全局向量集 kv 进行注意力交互
    - 仅对各模态的**全局 token**（CLS）做融合，轻量稳定；如需更强，可改为序列级交互

    结构：
    - 层归一化（pre-norm）+ MultiheadAttention + 残差
    - 层归一化 + 前馈网络（FFN）+ 残差
    """
    def __init__(self, d_model=256, nhead=8):
        super().__init__()                                          # 父类初始化
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)  # 多头注意力
        self.ffn = nn.Sequential(                                   # 前馈网络
            nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)                          # 第一路归一化
        self.norm2 = nn.LayerNorm(d_model)                          # 第二路归一化

    def fuse(self, q: torch.Tensor, kv: torch.Tensor):
        """
        执行一次跨模态融合：

        输入：
        - q : [B, 1, D] 查询模态的全局 token（某一模态）
        - kv: [B, K, D] 被融合模态的全局 token 集合（其他模态拼接）

        输出：
        - [B, 1, D] 融合后的查询模态全局 token
        """
        x = self.norm1(q)                         # 预归一化，稳定训练
        y = kv                                    # 被融合集合（不归一化也可）
        out, _ = self.attn(x, y, y)               # 多头注意力：q from x，k/v from y
        x = q + out                               # 残差连接
        x = x + self.ffn(self.norm2(x))           # FFN + 残差
        return x                                   # 返回融合后的全局 token


class CrossModalMixer(nn.Module):
    """
    多轮跨模态交互器：

    思路：
    - 维护多层 CrossModalBlock（layers>1 时表示多轮交互）
    - 每一轮中，对 text/image/audio/video 的全局向量，分别用“其余模态”的全局向量作为 kv 进行融合
      （即双向/多向并行融合）

    注意：
    - 为控制显存与复杂度，这里仅对全局 token 做融合
    - 如需更强交互，可扩展为“序列级共注意力”
    """
    def __init__(self, d_model=256, nhead=8, layers=2):
        super().__init__()                                          # 父类初始化
        self.blocks = nn.ModuleList(                                # 堆叠若干交互层
            [CrossModalBlock(d_model, nhead) for _ in range(layers)]
        )

    def forward(self, feats: Dict[str, torch.Tensor]):
        """
        输入：
        - feats: dict，包含四个键：'text','image','audio','video'，值均为 [B, D] 全局向量

        输出：
        - 与输入键相同的 dict，但每个向量已完成多轮跨模态融合
        """
        t = feats['text'].unsqueeze(1)      # [B,1,D] 文本全局向量
        i = feats['image'].unsqueeze(1)     # [B,1,D] 图像全局向量
        a = feats['audio'].unsqueeze(1)     # [B,1,D] 音频全局向量
        v = feats['video'].unsqueeze(1)     # [B,1,D] 视频全局向量

        for blk in self.blocks:             # 遍历每一轮交互
            t = blk.fuse(t, torch.cat([i, a, v], dim=1))  # 文本与（图像+音频+视频）交互
            i = blk.fuse(i, torch.cat([t, a, v], dim=1))  # 图像与（文本+音频+视频）交互
            a = blk.fuse(a, torch.cat([t, i, v], dim=1))  # 音频与（文本+图像+视频）交互
            v = blk.fuse(v, torch.cat([t, i, a], dim=1))  # 视频与（文本+图像+音频）交互

        return {                         # 压回 [B,D] 形状并返回
            'text': t.squeeze(1),
            'image': i.squeeze(1),
            'audio': a.squeeze(1),
            'video': v.squeeze(1),
        }


# =============================================================================
# 总模型：四模态编码 → 跨模态融合 → 线性投影（对比学习用） → 归一化
# =============================================================================

class MultiModalModel(nn.Module):
    """
    多模态对比学习主模型：

    结构：
    1) 四个单模态编码器：TextEncoder / ImageEncoder / AudioEncoder / VideoEncoder
    2) CrossModalMixer：对四个模态全局向量进行多轮跨模态融合
    3) 线性投影头：分别把四个模态投影到同一对比空间（可替换为 MLP）
    4) 可学习温度（logit_scale）：控制 InfoNCE 的对比分布“陡峭程度”

    前向输出：
    - 一个 dict，含四个模态归一化向量（'text','image','audio','video'）
    - 以及标量温度 'logit_scale'（已经做过 exp()，直接可用于相似度 logits 缩放）
    """
    def __init__(self, d_model=256, nhead=8, depth=4):
        super().__init__()                                          # 父类初始化
        # -------- 四个模态编码器 --------
        self.text_enc  = TextEncoder(d_model=d_model, nhead=nhead, depth=depth)  # 文本编码器
        self.img_enc   = ImageEncoder(d_model=d_model, nhead=nhead, depth=depth) # 图像编码器
        self.audio_enc = AudioEncoder(d_model=d_model, nhead=nhead, depth=depth) # 音频编码器
        self.vid_enc   = VideoEncoder(d_model=d_model, nhead=nhead, depth=depth) # 视频编码器

        # -------- 跨模态交互器 --------
        self.mixer = CrossModalMixer(d_model=d_model, nhead=nhead, layers=2)     # 两轮交互

        # -------- 线性投影头（可替换 MLP）--------
        self.proj_text  = nn.Linear(d_model, d_model, bias=False)   # 文本投影
        self.proj_image = nn.Linear(d_model, d_model, bias=False)   # 图像投影
        self.proj_audio = nn.Linear(d_model, d_model, bias=False)   # 音频投影
        self.proj_video = nn.Linear(d_model, d_model, bias=False)   # 视频投影

        # -------- 可学习温度（初始化为 CLIP 常用 1/0.07）--------
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))  # 标量参数（log 空间）

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        输入：
        - batch: dict，包含：
            - 'text_ids': [B, L] 文本 token id
            - 'image'   : [B, 3, H, W] 图像张量
            - 'audio'   : [B, 1, T] 音频波形或 1D 特征
            - 'video'   : [B, Tm, 3, H, W] 视频帧序列

        输出：
        - dict：
            - 'text'/'image'/'audio'/'video': 四个模态归一化后的对比向量 [B, D]
            - 'logit_scale': 温度标量（已 exp()）
        """
        t = self.text_enc(batch['text_ids'])     # [B,D] 文本全局向量
        i = self.img_enc(batch['image'])         # [B,D] 图像全局向量
        a = self.audio_enc(batch['audio'])       # [B,D] 音频全局向量
        v = self.vid_enc(batch['video'])         # [B,D] 视频全局向量

        fused = self.mixer({'text': t, 'image': i, 'audio': a, 'video': v})  # 跨模态融合

        # 线性投影 + L2 归一化（对比学习常规处理）
        t = F.normalize(self.proj_text(fused['text']),  dim=-1)     # [B,D]
        i = F.normalize(self.proj_image(fused['image']), dim=-1)    # [B,D]
        a = F.normalize(self.proj_audio(fused['audio']), dim=-1)    # [B,D]
        v = F.normalize(self.proj_video(fused['video']), dim=-1)    # [B,D]

        return {
            'text': t,
            'image': i,
            'audio': a,
            'video': v,
            'logit_scale': self.logit_scale.exp()                   # 返回 exp(logit_scale)
        }


# =============================================================================
# 对比学习损失：InfoNCE（双向）
# =============================================================================

def contrastive_loss(x: torch.Tensor, y: torch.Tensor, temperature: torch.Tensor):
    """
    计算 x 与 y 的双向 InfoNCE 对比损失（x→y 与 y→x 平均）：

    输入：
    - x, y: [B, D]，已做过 L2 归一化的向量（同一对比空间）
    - temperature: 标量温度（>0），用于缩放 logits

    过程：
    - 相似度矩阵 logits = x @ y^T * temperature，形状 [B, B]
    - label 为对角线索引 [0..B-1]，表示正样本在对角线上
    - CrossEntropy 分别计算行方向（x→y）与列方向（y→x），再取平均
    """
    logits = (x @ y.t()) * temperature           # [B,B] 余弦相似度（因已归一化）乘温度
    labels = torch.arange(x.size(0), device=x.device)  # [0..B-1]
    loss_xy = F.cross_entropy(logits, labels)    # x → y 的 CE 损失
    loss_yx = F.cross_entropy(logits.t(), labels)# y → x 的 CE 损失
    return 0.5 * (loss_xy + loss_yx)             # 双向平均


def total_contrastive_loss(feats: Dict[str, torch.Tensor], temp: torch.Tensor):
    """
    汇总多对模态的对比损失（等权）：

    输入：
    - feats: dict，包含四个模态的对比向量（已归一化）
    - temp : 温度标量

    当前包含 6 对：
    - T-I, T-A, T-V, I-V, I-A, A-V
    可按业务权重调整或删减。
    """
    t, i, a, v = feats['text'], feats['image'], feats['audio'], feats['video']  # 解包四模态
    loss = 0.0                                                                   # 初始化累计损失
    loss += contrastive_loss(t, i, temp)                                         # 文本-图像
    loss += contrastive_loss(t, a, temp)                                         # 文本-音频
    loss += contrastive_loss(t, v, temp)                                         # 文本-视频
    loss += contrastive_loss(i, v, temp)                                         # 图像-视频
    loss += contrastive_loss(i, a, temp)                                         # 图像-音频
    loss += contrastive_loss(a, v, temp)                                         # 音频-视频
    return loss / 6.0                                                            # 6 对等权平均


# =============================================================================
# 合成数据集：四模态 + 文本 token
# =============================================================================

class ToyDataset(torch.utils.data.Dataset):
    """
    合成数据集（用于演示与快速跑通）：

    生成规则：
    - 文本：均匀采样 token id，形状 [L]
    - 图像：高斯分布随机张量，形状 [3,H,W]
    - 音频：高斯分布随机张量，形状 [1,T]
    - 视频：高斯分布随机张量，形状 [Tm,3,H/2,W/2]

    注意：
    - 真实使用时请替换为你的数据读取逻辑（含预处理/归一化/对齐）
    """
    def __init__(self, num_samples=512, text_len=32, img_hw=128, audio_T=16000, vid_T=4, seed=42):
        super().__init__()                                          # 父类初始化
        g = torch.Generator().manual_seed(seed)                     # 固定随机种子
        self.data = []                                              # 用列表收集样本
        for _ in range(num_samples):                                # 循环生成样本
            text_ids = torch.randint(0, 30000, (text_len,), generator=g)       # 文本 token
            image    = torch.randn(3, img_hw, img_hw, generator=g)             # 图像张量
            audio    = torch.randn(1, audio_T, generator=g)                     # 音频张量
            video    = torch.randn(vid_T, 3, img_hw//2, img_hw//2, generator=g) # 视频张量
            self.data.append((text_ids, image, audio, video))       # 存入列表

    def __len__(self):
        return len(self.data)                                       # 返回样本数

    def __getitem__(self, idx):
        text_ids, image, audio, video = self.data[idx]              # 取对应样本
        return {                                                    # 返回字典，便于 DataLoader
            'text_ids': text_ids.long(),                            # 确保为 LongTensor
            'image': image.float(),                                 # 转 float32
            'audio': audio.float(),                                 # 转 float32
            'video': video.float(),                                 # 转 float32
        }


def collate_fn(batch):
    """
    简单的 batch 拼接函数（演示用）：

    - 文本：这里假设长度一致；真实项目需 pad 到同长，并传入 mask
    - 图像/音频/视频：直接按维度堆叠
    """
    text_ids = torch.stack([b['text_ids'] for b in batch], dim=0)   # [B,L]
    image    = torch.stack([b['image']    for b in batch], dim=0)   # [B,3,H,W]
    audio    = torch.stack([b['audio']    for b in batch], dim=0)   # [B,1,T]
    video    = torch.stack([b['video']    for b in batch], dim=0)   # [B,Tm,3,H,W]
    return {                                                        # 返回 batch 字典
        'text_ids': text_ids,
        'image': image,
        'audio': audio,
        'video': video
    }


# =============================================================================
# 训练 / 评估 / 保存 / 加载
# =============================================================================

def train_epoch(model, loader, optimizer, scaler, device):
    """
    单轮训练（遍历一个 DataLoader）：

    - 启用训练模式（model.train()）
    - 使用 AMP（autocast + GradScaler）进行混合精度（CUDA 自动开启）
    - 梯度清零 → 前向 → 计算损失 → 反向 → 优化器步进 → scaler 更新
    - 返回该轮的平均 loss
    """
    model.train()                                                   # 切换到训练模式
    total_loss = 0.0                                                # 累计损失
    for batch in loader:                                            # 遍历 mini-batch
        for k in batch: batch[k] = batch[k].to(device, non_blocking=True)  # 将数据搬到设备

        optimizer.zero_grad(set_to_none=True)                       # 梯度清零（高效置 None）
        # 选择 AMP 精度：CUDA 用 float16，CPU 则用 float32（避免 bfloat16 不可用）
        amp_dtype = torch.float16 if device.type == 'cuda' else torch.float32
        with torch.autocast(device_type=device.type, dtype=amp_dtype):      # 开启 autocast
            out = model(batch)                                      # 前向计算
            feats = {k: out[k] for k in ['text','image','audio','video']}   # 抽取四模态
            loss = total_contrastive_loss(feats, out['logit_scale'])         # 计算总对比损失

        scaler.scale(loss).backward()                               # AMP 下缩放反传
        scaler.step(optimizer)                                      # 优化器步进
        scaler.update()                                             # 更新 scaler 动态放大因子

        total_loss += loss.item()                                   # 累计损失值
    return total_loss / len(loader)                                 # 返回平均损失


@torch.no_grad()
def quick_eval(model, device, batch_size=8):
    """
    轻量级 sanity check（不做严格指标）：

    - 用小批量合成数据，前向生成四模态向量
    - 计算 Text-Image、Text-Video 相似度矩阵的对角 softmax 概率均值
    - 用于观察训练大致是否在朝“正确对齐”方向走（对角概率应高于随机）
    """
    model.eval()                                                    # 评估模式（关掉 dropout 等）
    dataset = ToyDataset(num_samples=batch_size)                    # 小规模临时数据
    batch = collate_fn([dataset[i] for i in range(batch_size)])     # 手工组 batch
    for k in batch: batch[k] = batch[k].to(device)                  # 搬到设备
    out = model(batch)                                              # 前向
    sims_ti = (out['text'] @ out['image'].t()).softmax(dim=-1).diag().mean().item()  # T-I 对角均值
    sims_tv = (out['text'] @ out['video'].t()).softmax(dim=-1).diag().mean().item()  # T-V 对角均值
    return {'TI_diag_prob': sims_ti, 'TV_diag_prob': sims_tv}       # 返回字典


def main():
    """
    主流程入口（重点讲解见文件末尾“main() 流程重点讲解”部分）：
    1) 固定随机种子；选择设备（CUDA/CPU）
    2) 构建数据集与 DataLoader（此处用合成数据演示）
    3) 实例化多模态模型、优化器、AMP GradScaler
    4) 训练若干轮：每轮结束做 quick_eval 观察对齐程度
    5) 保存 checkpoint（模型权重 + 配置）
    6) 重新加载模型并做一次 quick_eval 验证可复现
    """
    random.seed(0)                                  # Python 随机种子
    torch.manual_seed(0)                            # PyTorch 随机种子（CPU/GPU 算子尽可能复现）
    device = get_device()                           # 自动选择设备
    print(f"Using device: {device}")                # 打印设备信息

    # ---------- 超参数 ----------
    d_model   = 256                                 # 模型维度
    nhead     = 8                                   # 多头注意力头数
    depth     = 4                                   # Transformer 深度
    batch_size= 16                                  # 训练 batch 大小
    epochs    = 2                                   # 训练轮数（演示用设置较小）
    lr        = 2e-4                                # 学习率

    # ---------- 数据 ----------
    train_set = ToyDataset(                         # 构建合成训练集
        num_samples=256, text_len=32, img_hw=128, audio_T=16000, vid_T=4
    )
    train_loader = torch.utils.data.DataLoader(     # DataLoader（Windows 建议 num_workers=0）
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,                              # Windows 下多进程 dataloader 需小心，这里设 0
        pin_memory=(device.type=='cuda'),           # 若是 GPU，固定内存可略提速
        collate_fn=collate_fn                       # 使用我们自定义的拼接函数
    )

    # ---------- 模型 & 优化器 & AMP ----------
    model = MultiModalModel(d_model=d_model, nhead=nhead, depth=depth).to(device)  # 实例化模型并放到设备
    optimizer = torch.optim.AdamW(                 # AdamW（带权重衰减的 Adam）
        model.parameters(), lr=lr, weight_decay=0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))  # 仅在 CUDA 上启用 AMP scaler

    # ---------- 训练循环 ----------
    for e in range(1, epochs+1):                   # 逐轮训练
        loss = train_epoch(model, train_loader, optimizer, scaler, device)  # 训练一轮，返回平均损失
        eval_stats = quick_eval(model, device)     # 轻量评估（对角 softmax 概率）
        print(f"[Epoch {e}/{epochs}] loss={loss:.4f} "
              f" TI_diag_prob={eval_stats['TI_diag_prob']:.3f} "
              f" TV_diag_prob={eval_stats['TV_diag_prob']:.3f}")    # 打印训练与评估信息

    # ---------- 保存 ----------
    ckpt = {                                       # 构建 checkpoint 字典
        'model': model.state_dict(),               # 权重
        'config': {'d_model': d_model, 'nhead': nhead, 'depth': depth}  # 关键结构参数
    }
    torch.save(ckpt, "multimodal_contrastive_ckpt.pt")  # 序列化到磁盘
    print("已保存到 multimodal_contrastive_ckpt.pt")  # 提示保存成功

    # ---------- 加载 & 推理（sanity check） ----------
    loaded = MultiModalModel(**ckpt['config']).to(device)  # 以相同结构重新实例化
    loaded.load_state_dict(ckpt['model'])          # 加载权重
    loaded.eval()                                   # 切换评估模式
    stats = quick_eval(loaded, device)              # 再做一次轻量评估
    print(f"Reload Eval: TI_diag_prob={stats['TI_diag_prob']:.3f}, "
          f"TV_diag_prob={stats['TV_diag_prob']:.3f}")  # 打印加载后评估信息


# Python 脚本入口（直接运行本文件时触发 main）
if __name__ == "__main__":
    main()                                          # 调用主流程
