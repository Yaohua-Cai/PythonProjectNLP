# -*- coding: utf-8 -*-  # 指定源码文件编码为 UTF-8，确保中文注释正常
"""
多模态对比学习（Text/Image/Audio/Video）
- 文本特征：GPT2 Chinese (uer/gpt2-chinese-cluecorpussmall)
- 音频特征：Wav2Vec2 Base (facebook/wav2vec2-base-960h)
- 图像/视频：简化 ViT 风格编码（Patch→Transformer）
- 融合：基于 Transformer 的跨模态注意力（Cross-Modal Attention）
- 目标：InfoNCE 对比学习（T-I/T-A/T-V/I-A/I-V/A-V）

"""

# ============================== 标准库与类型提示 ==============================
import math                       # 数学函数库（对数、指数、三角函数等）
import random                     # 随机数库（设置随机种子等）
from typing import Dict           # 类型注解：字典类型

# ============================== PyTorch 相关 ==============================
import torch                      # PyTorch 主包
import torch.nn as nn             # 神经网络模块（层、容器）
import torch.nn.functional as F   # 函数式 API（激活、损失、归一化等）
from torch import amp             # 新式自动混合精度 GradScaler/Autocast 命名空间

# ============================== Transformers 相关 ==============================
from transformers import (        # 从 HuggingFace Transformers 导入所需模型与分词器接口
    AutoModel,                    # 通用自动模型加载器（用于 GPT-2 中文）
    AutoTokenizer,                # 通用自动分词器（本示例的合成数据未实际用到，留作真实数据接入）
    Wav2Vec2Model                 # Wav2Vec2 基础模型（音频特征抽取）
)

# ============================== 通用工具模块 ==============================
def get_device():
    """自动选择运行设备：有 CUDA 则用 GPU，否则用 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 返回 'cuda' 或 'cpu'

class PositionalEncoding(nn.Module):
    """正余弦位置编码：为序列特征注入位置信息。"""
    def __init__(self, d_model: int, max_len: int = 10000):  # d_model 为通道维，max_len 为最大序列长度
        super().__init__()                                    # 调用父类初始化
        pe = torch.zeros(max_len, d_model)                    # 构造 [L,D] 的位置编码矩阵
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # 位置索引 [L,1]
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / max_len))  # 频率缩放因子
        pe[:, 0::2] = torch.sin(pos * div)                    # 偶数通道用正弦
        pe[:, 1::2] = torch.cos(pos * div)                    # 奇数通道用余弦
        self.register_buffer("pe", pe.unsqueeze(0))           # 注册为 buffer，形状 [1,L,D]，不参与梯度

    def forward(self, x):                                      # x 形状 [B,L,D]
        return x + self.pe[:, :x.size(1), :]                  # 叠加对应长度的位置编码并返回

class AttentivePool(nn.Module):
    """注意力池化：用可学习查询 token 在序列上做软选择，得到全局向量。"""
    def __init__(self, d_model=768, nhead=8):                 # d_model 通道维，nhead 多头数
        super().__init__()                                     # 父类初始化
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))  # 可学习 query，形状 [1,1,D]
        nn.init.trunc_normal_(self.query, std=0.02)            # 截断正态初始化 query
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)  # 多头注意力层（批维在前）
        self.norm = nn.LayerNorm(d_model)                      # 层归一化，提升数值稳定

    def forward(self, seq):                                    # seq: [B,L,D] 序列特征
        q = self.query.expand(seq.size(0), -1, -1)             # 将查询 token 扩展到 batch 大小 -> [B,1,D]
        out, _ = self.attn(self.norm(q), self.norm(seq), self.norm(seq))  # q 与 seq 做注意力
        return out.squeeze(1)                                  # 压掉长度维 -> [B,D] 全局表示

class CLSHead(nn.Module):
    """为序列前面拼接一个可学习的 [CLS] token。"""
    def __init__(self, d_model: int):                          # d_model 通道维
        super().__init__()                                     # 父类初始化
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))    # 可学习 CLS，形状 [1,1,D]
        nn.init.trunc_normal_(self.cls, std=0.02)              # 截断正态初始化

    def forward(self, x):                                      # x: [B,L,D] 序列特征
        return torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)  # 拼接后 [B,1+L,D]

# ============================== 图像与视频编码模块 ==============================
class PatchEmbed2D(nn.Module):
    """图像 Patch 嵌入：使用 Conv2d（kernel=stride=patch）将图像切块并投影到 d_model。"""
    def __init__(self, in_ch=3, patch=16, d_model=256):        # in_ch 输入通道，patch 片大小，d_model 投影维
        super().__init__()                                      # 父类初始化
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)  # 卷积实现 patch 化

    def forward(self, x):                                       # x: [B,C,H,W]
        x = self.proj(x)                                        # 经过卷积 -> [B,D,h,w]
        x = x.flatten(2).transpose(1, 2).contiguous()           # 展平成序列 -> [B,N,D] 并确保内存连续
        return x                                                # 返回 patch 序列

class ImageEncoder(nn.Module):
    """简化版 ViT：Patch→(CLS+)Pos→Transformer→取 CLS 全局向量。"""
    def __init__(self, d_model=256, nhead=8, depth=4, patch=16, in_ch=3, max_patches=1024):  # 一些结构超参
        super().__init__()                                      # 父类初始化
        self.patch = PatchEmbed2D(in_ch=in_ch, patch=patch, d_model=d_model)  # 图像切片与投影
        self.cls_head = CLSHead(d_model)                        # 拼接 CLS
        self.pos = PositionalEncoding(d_model, max_len=max_patches+1)  # 位置编码（+1 预留 CLS）
        enc = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, batch_first=True)  # 单层 Transformer
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)  # 堆叠多个编码层

    def forward(self, img):                                     # img: [B,3,H,W]
        x = self.patch(img)                                     # 图像转序列 [B,N,D]
        x = self.cls_head(x)                                    # 拼 CLS -> [B,1+N,D]
        x = self.pos(x)                                         # 叠加位置编码
        x = self.encoder(x)                                     # Transformer 编码
        return x[:, 0, :]                                       # 取 CLS 位置 -> [B,D]

class VideoEncoder(nn.Module):
    """视频编码：逐帧做 PatchEmbed，再把所有帧的 patch 序列拼接为长序列，(CLS+)Pos→Transformer→取 CLS。"""
    def __init__(self, d_model=256, nhead=8, depth=4, patch=16, in_ch=3, max_tokens=4096):  # 结构超参
        super().__init__()                                      # 父类初始化
        self.frame_patch = PatchEmbed2D(in_ch=in_ch, patch=patch, d_model=d_model)  # 帧级 patch 编码
        self.cls_head = CLSHead(d_model)                        # 拼接 CLS
        self.pos = PositionalEncoding(d_model, max_len=max_tokens+1)  # 位置编码（+1 预留 CLS）
        enc = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, batch_first=True)  # 单层 Transformer
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)  # 堆叠编码层

    def forward(self, vid):                                     # vid: [B,T,3,H,W]，T 为帧数
        B, T, C, H, W = vid.shape                               # 解析维度
        vid = vid.reshape(B*T, C, H, W)                         # 合并时间维方便逐帧处理
        tokens = self.frame_patch(vid)                          # 每帧转为 patch 序列 [B*T, Np, D]
        Np = tokens.size(1)                                     # 每帧 patch 数
        tokens = tokens.reshape(B, T*Np, tokens.size(-1))       # 拆回批次，按时间拼成长序列 [B,T*Np,D]
        x = self.cls_head(tokens)                               # 拼接 CLS -> [B,1+T*Np,D]
        x = self.pos(x)                                         # 加位置编码
        x = self.encoder(x)                                     # Transformer 编码
        return x[:, 0, :]                                       # 取 CLS -> [B,D]

# ============================== 预训练文本与音频编码器 ==============================
class GPT2ChineseEncoder(nn.Module):
    """GPT-2 中文小模型特征抽取：最后隐层 + 注意力池化 → 线性投到统一维度。"""
    def __init__(self, out_dim=256, nhead_pool=8, model_name="gpt2-chinese-cluecorpussmall"):  # 输出维、注意力头数、模型名
        super().__init__()                                      # 父类初始化
        self.gpt2 = AutoModel.from_pretrained(model_name)       # 加载预训练 GPT-2 中文模型
        self.hidden = self.gpt2.config.hidden_size              # 记录隐藏维（通常 768）
        self.pool = AttentivePool(d_model=self.hidden, nhead=nhead_pool)  # 注意力池化器
        self.proj = nn.Linear(self.hidden, out_dim, bias=False) # 线性映射到统一 d_model

    def forward(self, token_ids, attention_mask=None):          # 输入 token_ids [B,L]，可选 attention_mask [B,L]
        out = self.gpt2(input_ids=token_ids, attention_mask=attention_mask, output_hidden_states=False)  # 前向
        seq = out.last_hidden_state                              # 取最后隐层序列 [B,L,H]
        feat = self.pool(seq)                                    # 注意力池化成全局向量 [B,H]
        return self.proj(feat)                                   # 线性映射到 out_dim -> [B,out_dim]

class Wav2Vec2Encoder(nn.Module):
    """Wav2Vec2 Base 特征抽取：最后隐层 + 注意力池化 → 线性投到统一维度。"""
    def __init__(self, out_dim=256, nhead_pool=8, model_name="wav2vec2-base-960h"):  # 输出维、注意力头数、模型名
        super().__init__()                                      # 父类初始化
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)  # 加载 Wav2Vec2 模型
        self.hidden = self.wav2vec2.config.hidden_size          # 记录隐藏维（通常 768）
        self.pool = AttentivePool(d_model=self.hidden, nhead=nhead_pool)  # 注意力池化器
        self.proj = nn.Linear(self.hidden, out_dim, bias=False) # 线性映射到统一 d_model

    def forward(self, wav):                                     # wav: [B,T]，float32，16kHz
        wav = wav.to(torch.float32)                             # 确保为 float32（AMP 下也强制）
        out = self.wav2vec2(wav, output_hidden_states=False)    # 前向，返回序列隐层
        seq = out.last_hidden_state                             # [B,L,H]，Wav2Vec2 的时间下采样输出
        feat = self.pool(seq)                                   # 注意力池化成全局向量 [B,H]
        return self.proj(feat)                                  # 线性映射到 out_dim -> [B,out_dim]

# ============================== 跨模态注意力融合模块 ==============================
class CrossModalBlock(nn.Module):
    """跨模态交互基本单元：q 与 kv 通过多头注意力融合 + FFN 残差。"""
    def __init__(self, d_model=256, nhead=8):                   # d_model 维度，nhead 头数
        super().__init__()                                      # 父类初始化
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)  # 多头注意力
        self.ffn = nn.Sequential(                               # 前馈网络（两层 MLP）
            nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)                      # 第一处层归一化
        self.norm2 = nn.LayerNorm(d_model)                      # 第二处层归一化

    def fuse(self, q, kv):                                      # q: [B,1,D]；kv: [B,K,D]
        x = self.norm1(q)                                       # 归一化查询
        y = self.norm1(kv)                                      # 归一化键值
        out, _ = self.attn(x, y, y)                             # 注意力：q=x, k=y, v=y
        x = q + out                                             # 残差连接
        x = x + self.ffn(self.norm2(x))                         # FFN + 残差
        return x                                                # 返回融合后的查询

class CrossModalMixer(nn.Module):
    """多轮跨模态融合器：对 T/I/A/V 四个全局向量做双向/多向交互（仅在全局 token 上融合）。"""
    def __init__(self, d_model=256, nhead=8, layers=2):         # d_model、头数、交互层数
        super().__init__()                                      # 父类初始化
        self.blocks = nn.ModuleList([CrossModalBlock(d_model, nhead) for _ in range(layers)])  # 堆叠多层

    def forward(self, feats: Dict[str, torch.Tensor]):          # feats 中包含 'text','image','audio','video': [B,D]
        t = feats['text'].unsqueeze(1)                          # 扩展序列维 -> [B,1,D]
        i = feats['image'].unsqueeze(1)                         # 同上
        a = feats['audio'].unsqueeze(1)                         # 同上
        v = feats['video'].unsqueeze(1)                         # 同上
        for blk in self.blocks:                                 # 多轮交互
            t = blk.fuse(t, torch.cat([i, a, v], dim=1))        # 文本融合来自 图像+音频+视频 的信息
            i = blk.fuse(i, torch.cat([t, a, v], dim=1))        # 图像融合来自 文本+音频+视频 的信息
            a = blk.fuse(a, torch.cat([t, i, v], dim=1))        # 音频融合来自 文本+图像+视频 的信息
            v = blk.fuse(v, torch.cat([t, i, a], dim=1))        # 视频融合来自 文本+图像+音频 的信息
        return {                                                # 压回 [B,D] 并返回字典
            'text': t.squeeze(1),
            'image': i.squeeze(1),
            'audio': a.squeeze(1),
            'video': v.squeeze(1)
        }

# ============================== 总模型（特征提取 → 融合 → 投影） ==============================
class MultiModalModel(nn.Module):
    """
    主模型：
    - 文本：GPT2 中文 + 注意力池化 → 线性到 d_model
    - 音频：Wav2Vec2 Base + 注意力池化 → 线性到 d_model
    - 图像/视频：简化 ViT 编码到 d_model
    - 融合：CrossModalMixer（多轮跨模态注意力）
    - 投影：四个线性头 + L2 归一化（对比学习空间）
    - 温度：可学习 logit_scale（初始化到 1/0.07）
    """
    def __init__(self, d_model=256, nhead=8, depth=4):          # 统一 d_model、头数、Transformer 深度
        super().__init__()                                      # 父类初始化
        self.text_enc  = GPT2ChineseEncoder(out_dim=d_model, nhead_pool=8)          # 文本编码器（预训练）
        self.audio_enc = Wav2Vec2Encoder(out_dim=d_model, nhead_pool=8)             # 音频编码器（预训练）
        self.img_enc   = ImageEncoder(d_model=d_model, nhead=nhead, depth=depth)     # 图像编码器（简化）
        self.vid_enc   = VideoEncoder(d_model=d_model, nhead=nhead, depth=depth)     # 视频编码器（简化）
        self.mixer     = CrossModalMixer(d_model=d_model, nhead=nhead, layers=2)     # 跨模态融合器
        self.proj_text  = nn.Linear(d_model, d_model, bias=False)                    # 文本投影到对比空间
        self.proj_image = nn.Linear(d_model, d_model, bias=False)                    # 图像投影
        self.proj_audio = nn.Linear(d_model, d_model, bias=False)                    # 音频投影
        self.proj_video = nn.Linear(d_model, d_model, bias=False)                    # 视频投影
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))              # 可学习温度（log 空间）

    def forward(self, batch: Dict[str, torch.Tensor]):           # batch 包含 text_ids/text_attn/image/audio/video
        t = self.text_enc(batch['text_ids'], attention_mask=batch.get('text_attn'))  # 文本全局向量 [B,D]
        i = self.img_enc(batch['image'])                         # 图像全局向量 [B,D]
        audio = batch['audio']                                   # 取出音频
        if audio.dim() == 3 and audio.size(1) == 1:              # 若形如 [B,1,T]，挤掉通道维
            audio = audio.squeeze(1)                             # 变为 [B,T]
        a = self.audio_enc(audio)                                # 音频全局向量 [B,D]
        v = self.vid_enc(batch['video'])                         # 视频全局向量 [B,D]
        fused = self.mixer({'text': t, 'image': i, 'audio': a, 'video': v})          # 跨模态融合
        t = F.normalize(self.proj_text(fused['text']),  dim=-1)  # 文本投影 + 归一化
        i = F.normalize(self.proj_image(fused['image']), dim=-1) # 图像投影 + 归一化
        a = F.normalize(self.proj_audio(fused['audio']), dim=-1) # 音频投影 + 归一化
        v = F.normalize(self.proj_video(fused['video']), dim=-1) # 视频投影 + 归一化
        return {'text': t, 'image': i, 'audio': a, 'video': v, 'logit_scale': self.logit_scale.exp()}  # 返回字典

# ============================== 对比学习损失（InfoNCE 双向） ==============================
def contrastive_loss(x, y, temperature):                        # x,y:[B,D] 已归一化；temperature 标量
    logits = (x @ y.t()) * temperature                          # 余弦相似度（因已归一化）乘温度 -> [B,B]
    labels = torch.arange(x.size(0), device=x.device)            # 对角为正样本，标签 [0..B-1]
    return 0.5 * (                                              # 双向平均（x→y 与 y→x）
        F.cross_entropy(logits, labels) +                        # 行方向 CE
        F.cross_entropy(logits.t(), labels)                      # 列方向 CE
    )

def total_contrastive_loss(feats: Dict[str, torch.Tensor], temp: torch.Tensor):  # 汇总 6 对模态损失
    t, i, a, v = feats['text'], feats['image'], feats['audio'], feats['video']    # 解包四模态
    loss = 0                                                                      # 初始化
    loss += contrastive_loss(t, i, temp)                                          # 文本-图像
    loss += contrastive_loss(t, a, temp)                                          # 文本-音频
    loss += contrastive_loss(t, v, temp)                                          # 文本-视频
    loss += contrastive_loss(i, a, temp)                                          # 图像-音频
    loss += contrastive_loss(i, v, temp)                                          # 图像-视频
    loss += contrastive_loss(a, v, temp)                                          # 音频-视频
    return loss / 6.0                                                             # 等权平均 6 对

# ============================== 合成数据集（演示用） ==============================
class ToyDataset(torch.utils.data.Dataset):
    """
    合成四模态数据以跑通流程：
    - 文本：随机 token id（默认中文词表规模 21128）
    - 图像/音频/视频：高斯随机张量
    """
    def __init__(self, num_samples=128, text_len=32, img_hw=128, audio_T=16000, vid_T=4,
                 text_vocab_size=21128, seed=42):                # 数据规模与形状超参
        super().__init__()                                       # 父类初始化
        g = torch.Generator().manual_seed(seed)                  # 固定随机种子以复现
        self.data = []                                           # 用列表存样本
        for _ in range(num_samples):                             # 循环生成样本
            token_ids = torch.randint(0, text_vocab_size, (text_len,), generator=g)    # 文本 token [L]
            image = torch.randn(3, img_hw, img_hw, generator=g)                         # 图像 [3,H,W]
            audio = torch.randn(audio_T, generator=g)                                   # 音频 [T] （Wav2Vec2 用 [B,T]）
            video = torch.randn(vid_T, 3, img_hw//2, img_hw//2, generator=g)           # 视频 [Tm,3,H/2,W/2]
            self.data.append((token_ids, image, audio, video))                          # 追加到数据列表

    def __len__(self):                                            # 返回样本总数
        return len(self.data)

    def __getitem__(self, idx):                                   # 返回第 idx 个样本
        token_ids, image, audio, video = self.data[idx]           # 解包
        return {                                                  # 组织为字典（与模型前向期望一致）
            'text_ids': token_ids.long(),                         # 文本 token [L]
            'text_attn': torch.ones_like(token_ids).long(),       # 简单全1 mask（真实需按 pad 生成）
            'image': image.float(),                               # 图像 [3,H,W]
            'audio': audio.float(),                               # 音频 [T]
            'video': video.float(),                               # 视频 [Tm,3,H/2,W/2]
        }

def collate_fn(batch):                                            # 自定义批处理函数
    text_ids  = torch.stack([b['text_ids']  for b in batch], 0)   # 堆叠文本 -> [B,L]
    text_attn = torch.stack([b['text_attn'] for b in batch], 0)   # 堆叠 mask -> [B,L]
    image     = torch.stack([b['image']     for b in batch], 0)   # 堆叠图像 -> [B,3,H,W]
    audio     = torch.stack([b['audio']     for b in batch], 0)   # 堆叠音频 -> [B,T]（示例等长）
    video     = torch.stack([b['video']     for b in batch], 0)   # 堆叠视频 -> [B,Tm,3,H,W]
    return {                                                       # 返回批字典
        'text_ids': text_ids,
        'text_attn': text_attn,
        'image': image,
        'audio': audio,
        'video': video
    }

# ============================== 训练 / 轻量评估 / 保存 / 加载 ==============================
def train_epoch(model, loader, optimizer, scaler, device):         # 单轮训练
    model.train()                                                  # 切换训练模式
    total = 0.0                                                    # 累计损失
    for batch in loader:                                           # 遍历每个 mini-batch
        for k in batch: batch[k] = batch[k].to(device, non_blocking=True)  # 数据搬到设备
        optimizer.zero_grad(set_to_none=True)                      # 梯度置 None（更高效）
        dtype = torch.float16 if device.type == 'cuda' else torch.float32  # AMP 精度选择
        with torch.autocast(device_type=device.type, dtype=dtype):         # 开启自动混合精度
            out = model(batch)                                     # 前向计算（特征提取+融合+投影）
            feats = {k: out[k] for k in ['text','image','audio','video']}  # 抽取四模态向量
            loss = total_contrastive_loss(feats, out['logit_scale'])       # 计算对比学习总损失
        scaler.scale(loss).backward()                               # AMP 缩放反向传播
        scaler.step(optimizer)                                      # 优化器步进
        scaler.update()                                             # 更新缩放器
        total += loss.item()                                        # 累加损失
    return total / len(loader)                                      # 返回平均损失

@torch.no_grad()                                                    # 评估不需要梯度
def quick_eval(model, device, batch_size=8):                        # 轻量级 sanity check
    model.eval()                                                    # 切换评估模式
    ds = ToyDataset(num_samples=batch_size)                         # 小数据集
    bt = collate_fn([ds[i] for i in range(batch_size)])             # 手工组一个 batch
    for k in bt: bt[k] = bt[k].to(device)                           # 搬到设备
    out = model(bt)                                                 # 前向
    ti = (out['text'] @ out['image'].t()).softmax(-1).diag().mean().item()  # T-I 对角 softmax 概率均值
    tv = (out['text'] @ out['video'].t()).softmax(-1).diag().mean().item()  # T-V 对角 softmax 概率均值
    return {'TI_diag_prob': ti, 'TV_diag_prob': tv}                 # 返回两个简易指标

def main():                                                         # 主流程入口
    random.seed(0); torch.manual_seed(0)                            # 固定随机种子
    device = get_device()                                           # 自动选择设备
    print(f"Using device: {device}")                                # 打印设备信息

    d_model = 256; nhead = 8; depth = 4                             # 模型结构超参
    batch_size = 8; epochs = 2; lr = 2e-4                           # 训练超参（示例设置较小以快速跑通）

    train_set = ToyDataset(num_samples=64, text_len=32, img_hw=128, audio_T=16000, vid_T=4)  # 构造训练集
    train_loader = torch.utils.data.DataLoader(                     # DataLoader
        train_set, batch_size=batch_size, shuffle=True, num_workers=0,  # Windows 建议 0 worker 简化
        pin_memory=(device.type=='cuda'), collate_fn=collate_fn)    # GPU 时 pin_memory=True 略提速

    model = MultiModalModel(d_model=d_model, nhead=nhead, depth=depth).to(device)  # 实例化模型并放设备
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)    # AdamW 优化器
    scaler = amp.GradScaler('cuda', enabled=(device.type == 'cuda'))               # 新式 GradScaler 构造

    for e in range(1, epochs+1):                                    # 训练若干轮
        loss = train_epoch(model, train_loader, optimizer, scaler, device)         # 训练一轮
        ev = quick_eval(model, device)                              # 轻量评估（对角概率）
        print(f"[Epoch {e}/{epochs}] loss={loss:.4f} TI_diag_prob={ev['TI_diag_prob']:.3f} TV_diag_prob={ev['TV_diag_prob']:.3f}")  # 打印信息

    ckpt = {'model': model.state_dict(), 'config': {'d_model': d_model, 'nhead': nhead, 'depth': depth}}  # 组装 checkpoint
    torch.save(ckpt, "mm_gpt2cn_w2v2_ckpt.pt")                      # 保存到磁盘
    print("已保存到 mm_gpt2cn_w2v2_ckpt.pt")                     # 打印保存成功

    loaded = MultiModalModel(**ckpt['config']).to(device)           # 用相同配置重新实例化模型
    loaded.load_state_dict(ckpt['model'])                           # 加载权重
    loaded.eval()                                                   # 评估模式
    ev2 = quick_eval(loaded, device)                                # 再做一次轻量评估
    print(f"Reload Eval: TI_diag_prob={ev2['TI_diag_prob']:.3f} TV_diag_prob={ev2['TV_diag_prob']:.3f}")  # 打印评估结果

# 脚本入口：直接运行本文件时执行 main()
if __name__ == "__main__":
    main()                                                          # 启动主流程
