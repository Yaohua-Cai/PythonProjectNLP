# =========================
# Minimal BERT Encoder in PyTorch
# 重点：自注意力、多头注意力、FFN、残差、LayerNorm、三重嵌入、Padding Mask
# =========================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)                          # 固定随机种子，便于复现
torch.set_printoptions(precision=4, sci_mode=False)

# ---------- 嵌入层：Token + Position + Segment(=TokenType) ----------
class BertEmbeddings(nn.Module):
    """
    三重嵌入并相加：token_embedding + position_embedding + segment_embedding
    - token_emb: 将 token id -> 向量
    - position_emb: 绝对位置编码（可学习）
    - token_type_emb: 句子片段 A/B（用于句对任务；单句时全 0）
    """
    def __init__(self, vocab_size: int, hidden_size: int, max_len: int = 512, type_vocab_size: int = 2, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)       # [B,L] -> [B,L,H]
        self.pos_emb   = nn.Embedding(max_len, hidden_size)          # [L]   -> [L,H]
        self.type_emb  = nn.Embedding(type_vocab_size, hidden_size)  # [B,L] -> [B,L,H]
        self.layer_norm = nn.LayerNorm(hidden_size)                  # 规范化，稳定训练
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None):
        """
        input_ids:     [B,L]  每个位置是词表 id
        token_type_ids:[B,L]  每个位置是片段 id（0/1），单句可全 0；为空则自动置 0
        返回:
          x: [B,L,H]  三重嵌入相加、LayerNorm 和 Dropout 后的表示
        """
        B, L = input_ids.shape                                        # 批大小与序列长度
        device = input_ids.device

        # 1) 词向量
        tok = self.token_emb(input_ids)                                # [B,L,H]

        # 2) 位置向量：0..L-1
        pos_ids = torch.arange(L, device=device)                       # [L]
        pos = self.pos_emb(pos_ids).unsqueeze(0).expand(B, L, -1)      # [B,L,H]

        # 3) 片段向量：若未提供，置 0
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)               # [B,L]
        typ = self.type_emb(token_type_ids)                            # [B,L,H]

        # 4) 三者相加 + 规范化 + 随机失活
        x = tok + pos + typ                                            # [B,L,H]
        x = self.layer_norm(x)                                         # [B,L,H]
        x = self.dropout(x)                                            # [B,L,H]
        return x


# ---------- 缩放点积注意力（单头） ----------
class ScaledDotProductAttention(nn.Module):
    """
    单头注意力：softmax((QK^T)/sqrt(d)) V
    注意：BERT 是双向的，不使用因果掩码，但需要 padding mask 屏蔽 [PAD]
    """
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        Q,K,V: [B,h,L,d_head]  注意多头前已被拆分
        attn_mask: [B,1,1,L] 或 [B,1,L,L] 的 0/1 mask（1=可见，0=屏蔽），会加到分数上
        返回:
          context: [B,h,L,d_head]
          attn:    [B,h,L,L] 注意力权重矩阵（便于可视化）
        """
        d_head = Q.size(-1)                                           # 每头维度
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_head)        # [B,h,L,L] = QK^T/sqrt(d)
        if attn_mask is not None:
            # 将不可见位置的分数置为 -inf，softmax 后 ~ 0
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)                              # [B,h,L,L]
        context = attn @ V                                            # [B,h,L,d_head]
        return context, attn


# ---------- 多头自注意力（MHSA） ----------
class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力：输入 X -> 线性映射成 Q/K/V -> 拆头 -> 单头注意力 -> 拼头 -> 输出线性层
    """
    def __init__(self, hidden_size: int, num_heads: int, attn_dropout: float = 0.1, proj_dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size 必须被 num_heads 整除"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_head = hidden_size // num_heads                         # 每头维度

        # 三个线性层：映射到 Q/K/V 的串联空间（总维不变）
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)

        # 单头注意力模块（可复用）
        self.attn_core = ScaledDotProductAttention()

        # 拼接后再做一次线性映射（通常称 W_o）
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)

        # 两处 dropout：注意力权重与输出投影
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B,L,H] -> [B,h,L,d_head]
        """
        B, L, H = x.shape
        x = x.view(B, L, self.num_heads, self.d_head)                  # [B,L,h,d_head]
        x = x.permute(0, 2, 1, 3).contiguous()                         # [B,h,L,d_head]
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B,h,L,d_head] -> [B,L,H]
        """
        B, h, L, Dh = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()                         # [B,L,h,d_head]
        x = x.view(B, L, h * Dh)                                       # [B,L,H]
        return x

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, need_weights: bool = False):
        """
        x: [B,L,H]
        attn_mask: [B,1,1,L] 或 [B,1,L,L]，1=可见，0=屏蔽（BERT 常用 [B,1,1,L] 的 padding mask）
        need_weights: 是否返回注意力矩阵方便可视化
        返回:
          out: [B,L,H]
          attn(可选): [B,h,L,L]
        """
        # 1) 线性映射到 Q/K/V 空间
        Q = self.W_q(x)                                                # [B,L,H]
        K = self.W_k(x)                                                # [B,L,H]
        V = self.W_v(x)                                                # [B,L,H]

        # 2) 拆成多头形状
        Qh = self._split_heads(Q)                                      # [B,h,L,d_head]
        Kh = self._split_heads(K)                                      # [B,h,L,d_head]
        Vh = self._split_heads(V)                                      # [B,h,L,d_head]

        # 3) 单头注意力（带 padding mask）
        context, attn = self.attn_core(Qh, Kh, Vh, attn_mask)          # context:[B,h,L,d_head]

        # 4) 权重 dropout（可选）
        attn = self.attn_dropout(attn)

        # 5) 合并多头并线性投影
        out = self._merge_heads(context)                               # [B,L,H]
        out = self.W_o(out)                                            # [B,L,H]
        out = self.proj_dropout(out)                                   # [B,L,H]

        if need_weights:
            return out, attn
        return out


# ---------- 前馈网络（逐位置两层 MLP） ----------
class FeedForward(nn.Module):
    """
    BERT 风格 FFN：线性 -> GELU -> 线性 -> Dropout
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)           # 扩维
        self.fc2 = nn.Linear(intermediate_size, hidden_size)           # 回到原维
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)                                                # [B,L,inter]
        x = F.gelu(x)                                                  # GELU 激活（BERT 使用）
        x = self.fc2(x)                                                # [B,L,H]
        x = self.dropout(x)                                            # 正则
        return x


# ---------- BertEncoderLayer（注意力 + FFN，均为 Pre-LN 残差结构） ----------
class BertEncoderLayer(nn.Module):
    """
    单层 BERT 编码器块：
      x -> LN -> MHSA -> 残差
      x -> LN -> FFN  -> 残差
    注：原始 BERT 论文用的是 Post-LN（Sublayer 后再 LN），
        现代实现常用 Pre-LN（先 LN 再 Sublayer），训练更稳；两者思想一致。
    """
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, attn_dropout=0.1, resid_dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)                           # 注意力子层前置 LN
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, attn_dropout=attn_dropout, proj_dropout=resid_dropout)
        self.ln2 = nn.LayerNorm(hidden_size)                           # FFN 子层前置 LN
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout=resid_dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, need_attn=False):
        # 子层 1：LN -> MHSA（双向，带 padding 掩码）-> 残差
        h = self.ln1(x)                                                # [B,L,H]
        a_out = self.attn(h, attn_mask, need_weights=need_attn)        # [B,L,H] (+ attn)
        if need_attn:
            a_out, attn_weights = a_out                                # 取权重
        x = x + a_out                                                  # 残差

        # 子层 2：LN -> FFN -> 残差
        h = self.ln2(x)                                                # [B,L,H]
        f_out = self.ffn(h)                                            # [B,L,H]
        x = x + f_out                                                  # 残差

        if need_attn:
            return x, attn_weights
        return x


# ---------- BertEncoder（堆叠多层） ----------
class BertEncoder(nn.Module):
    """
    堆叠 N 层编码器，每层都是 BertEncoderLayer
    """
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, intermediate_size: int, attn_dropout=0.1, resid_dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BertEncoderLayer(hidden_size, num_heads, intermediate_size, attn_dropout, resid_dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)                          # 末端 LN（可选）

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, need_all_attn=False):
        """
        x: [B,L,H] 来自嵌入
        attn_mask: [B,1,1,L] 或 [B,1,L,L] 的 padding 掩码
        need_all_attn: 若 True，收集每层注意力权重
        返回:
          x: 编码后的序列表示 [B,L,H]
          attn_list(可选): List[[B,h,L,L], ...]
        """
        attn_list = []
        for layer in self.layers:
            if need_all_attn:
                x, attn_w = layer(x, attn_mask, need_attn=True)
                attn_list.append(attn_w)
            else:
                x = layer(x, attn_mask, need_attn=False)
        x = self.ln_f(x)
        if need_all_attn:
            return x, attn_list
        return x


# ---------- BertModel（整合：Embedding -> Encoder -> Pooler） ----------
class BertModel(nn.Module):
    """
    最小 BERT：
      - 三重嵌入（token/pos/segment）
      - 编码器堆叠（双向注意力）
      - pooler: 取 [CLS] 向量过线性+Tanh（可用于分类/NSP）
    """
    def __init__(self, vocab_size: int, hidden_size: int = 256, num_layers: int = 4,
                 num_heads: int = 4, intermediate_size: int = 1024, max_len: int = 512,
                 type_vocab_size: int = 2, attn_dropout=0.1, resid_dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_len, type_vocab_size, dropout=emb_dropout)
        self.encoder = BertEncoder(num_layers, hidden_size, num_heads, intermediate_size, attn_dropout, resid_dropout)
        # pooler：取 CLS 位置（第 0 个 token）的隐藏向量 -> 线性 -> tanh
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def build_padding_mask(self, input_ids: torch.Tensor, pad_id: int = 0):
        """
        构建 padding mask：1=可见，0=屏蔽；形状 [B,1,1,L]，便于广播到 [B,h,L,L]
        BERT 是双向注意力，mask 仅用于“屏蔽 PAD”，不限制未来。
        """
        # 1) 有效位置为 True（input_ids != pad_id）
        valid = (input_ids != pad_id).unsqueeze(1).unsqueeze(2)        # [B,1,1,L]
        # 2) 转为整型 0/1
        return valid.to(dtype=torch.int)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None,
                pad_id: int = 0, need_attn: bool = False):
        """
        input_ids:     [B,L]  token ids
        token_type_ids:[B,L]  片段 ids（句对时 0/1；单句可全 0）
        pad_id:        词表中 [PAD] 的 id（默认 0）
        need_attn:     是否返回每层注意力权重
        返回:
          seq_output:  [B,L,H]  序列每个位置的编码向量（供标注/QA等）
          pooled:      [B,H]    取 CLS 的 Tanh 后表示（供分类/NSP等）
          attn_list(可选): List[[B,h,L,L], ...]
        """
        # 1) 构建嵌入
        x = self.embeddings(input_ids, token_type_ids)                  # [B,L,H]

        # 2) 构建 padding 掩码（双向注意力仅屏蔽 PAD）
        attn_mask = self.build_padding_mask(input_ids, pad_id)          # [B,1,1,L]

        # 3) 过编码器堆叠
        if need_attn:
            seq_output, attn_list = self.encoder(x, attn_mask, need_all_attn=True)
        else:
            seq_output = self.encoder(x, attn_mask, need_all_attn=False)

        # 4) pooler：拿 CLS（第 0 位）的表示做线性+Tanh
        cls = seq_output[:, 0, :]                                       # [B,H]
        pooled = self.pooler(cls)                                       # [B,H]

        if need_attn:
            return seq_output, pooled, attn_list
        return seq_output, pooled


# =========================
# 最小可运行 Demo
# =========================
if __name__ == "__main__":
    # 超小配置：便于 CPU 跑通
    vocab_size = 30522          # 与 BERT 同风格（随便设个合理数）
    hidden_size = 128
    num_layers = 2
    num_heads  = 4
    intermediate_size = 4 * hidden_size
    max_len = 64
    pad_id = 0

    model = BertModel(vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_len=max_len)

    # 构造一个玩具 batch：B=2, L=10
    B, L = 2, 10
    # 构造 token ids（0 作为 PAD），后面几位用 0 模拟 padding
    input_ids = torch.tensor([
        [101, 2009, 2003, 1037, 7953, 102, 0, 0, 0, 0],   # 样本1（[CLS]=101，[SEP]=102）
        [101, 1045, 2293, 2023, 3185, 102, 0, 0, 0, 0],   # 样本2
    ], dtype=torch.long)
    # 片段 ids（单句任务 -> 全 0）
    token_type_ids = torch.zeros_like(input_ids)

    # 前向：拿到序列输出、池化向量和每层注意力权重
    seq_out, pooled, attn_list = model(input_ids, token_type_ids, pad_id=pad_id, need_attn=True)

    print("输入形状:", input_ids.shape)                  # [2,10]
    print("序列输出形状:", seq_out.shape)                 # [2,10,128]
    print("池化向量形状:", pooled.shape)                  # [2,128]
    print("层数/注意力形状:", len(attn_list), attn_list[0].shape)  # 2 层, [2,4,10,10]
