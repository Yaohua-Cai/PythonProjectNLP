# =========================
# Minimal GPT in PyTorch
# 重点讲解：MHSA(含因果mask) + FFN + Residual + LayerNorm + Embedding + Generate
# =========================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 为了可复现
torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

# ---------- 工具：构造因果掩码（上三角为 -inf，阻止看到未来） ----------
def build_causal_mask(seq_len: int, device=None, dtype=None):
    """
    返回形状 [1, 1, L, L] 的因果掩码（broadcast 到 [B,h,L,L] 使用）：
      对 j>i 的位置（未来）置为 -inf；对 j<=i 的位置置为 0。
    """
    # 先构造上三角（不含对角）为 1 的矩阵，再取 bool
    tri = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    # True 的地方（未来位置）置为 -inf，False 的地方置 0
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
    mask = mask.masked_fill(tri, float("-inf"))  # [L, L]
    # 扩展出 batch 和 head 维度以便广播
    return mask.view(1, 1, seq_len, seq_len)     # [1,1,L,L]


# ---------- 多头自注意力（含因果掩码） ----------
class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力（Masked）：Q=K=V=输入X的线性投影。
    - 输入：x:[B,L,d_model]
    - 输出：out:[B,L,d_model]，并可返回注意力权重便于可视化
    """
    def __init__(self, d_model: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # 单头维度

        # 线性层把 d_model 投影到 Q/K/V 三个空间（总维度仍是 d_model，等价于 h 个 d_head 串在一起）
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 头拼接后的线性映射
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # dropout：注意力权重与输出投影处
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def _split_heads(self, x: torch.Tensor):
        """
        把 [B,L,d_model] -> [B,h,L,d_head] 以便做多头并行。
        """
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.d_head)  # [B,L,h,d_head]
        x = x.permute(0, 2, 1, 3).contiguous()         # [B,h,L,d_head]
        return x

    def _merge_heads(self, x: torch.Tensor):
        """
        把 [B,h,L,d_head] -> [B,L,d_model]
        """
        B, h, L, Dh = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()         # [B,L,h,d_head]
        x = x.view(B, L, h * Dh)                       # [B,L,d_model]
        return x

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, need_weights: bool = False):
        """
        x: [B,L,d_model]
        attn_mask: [1,1,L,L] 或 [B,1,L,L]（因果掩码已是 -inf/0）
        返回:
          out: [B,L,d_model]
          weights(可选): [B,h,L,L]
        """
        B, L, D = x.shape

        # 1) 线性投影得到 Q、K、V（共享输入 X）
        Q = self.W_q(x)  # [B,L,d_model]
        K = self.W_k(x)  # [B,L,d_model]
        V = self.W_v(x)  # [B,L,d_model]

        # 2) 拆成多头形状
        Q = self._split_heads(Q)  # [B,h,L,d_head]
        K = self._split_heads(K)  # [B,h,L,d_head]
        V = self._split_heads(V)  # [B,h,L,d_head]

        # 3) 注意力分数：QK^T / sqrt(d_head)
        #    K^T: [B,h,d_head,L]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,h,L,L]

        # 4) 加上因果掩码：未来位置得分置为 -inf（softmax 后 ~0）
        if attn_mask is not None:
            # attn_mask 形状兼容 [1,1,L,L] -> 广播到 [B,h,L,L]
            scores = scores + attn_mask

        # 5) softmax 得到注意力权重；可选 dropout
        attn = F.softmax(scores, dim=-1)             # [B,h,L,L]
        attn = self.attn_dropout(attn)

        # 6) 加权求和：αV
        context = attn @ V                            # [B,h,L,d_head]

        # 7) 合并多头并线性投影
        out = self._merge_heads(context)              # [B,L,d_model]
        out = self.W_o(out)                           # [B,L,d_model]
        out = self.proj_dropout(out)

        if need_weights:
            return out, attn
        return out


# ---------- 前馈网络（逐位置两层 MLP） ----------
class FeedForward(nn.Module):
    """
    GPT 风格 MLP：线性 -> GELU -> 线性 -> dropout
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 扩维
        self.fc2 = nn.Linear(d_ff, d_model)  # 回到 d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)            # [B,L,d_ff]
        x = F.gelu(x)              # 非线性激活（GPT 常用 GELU）
        x = self.fc2(x)            # [B,L,d_model]
        x = self.dropout(x)
        return x


# ---------- GPT Block（Pre-LN：LN->Sublayer->残差） ----------
class GPTBlock(nn.Module):
    """
    单个解码器块：Pre-LN + Masked-MHSA + 残差 + Pre-LN + FFN + 残差
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)                          # 对注意力子层做 Pre-LN
        self.attn = MultiHeadSelfAttention(d_model, num_heads, attn_dropout=attn_dropout, proj_dropout=resid_dropout)
        self.ln2 = nn.LayerNorm(d_model)                          # 对 MLP 子层做 Pre-LN
        self.ffn = FeedForward(d_model, d_ff, dropout=resid_dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, need_attn=False):
        # 子层1：LN -> MHSA -> 残差
        h = self.ln1(x)                                           # 归一化（稳定训练）
        a_out = self.attn(h, attn_mask, need_weights=need_attn)   # 注意力（掩码保证因果性）
        if need_attn:
            a_out, attn_weights = a_out
        x = x + a_out                                             # 残差：保留原始信息 + 新信息

        # 子层2：LN -> FFN -> 残差
        h = self.ln2(x)
        f_out = self.ffn(h)
        x = x + f_out

        if need_attn:
            return x, attn_weights
        return x


# ---------- GPT 模型骨架 ----------
class GPT(nn.Module):
    """
    最小 GPT：
    - 词嵌入（token embedding）
    - 位置嵌入（learnable，GPT-2 风格）
    - N 层解码器块（Masked MHSA + FFN + 残差 + LayerNorm）
    - 语言建模头（共享词嵌入权重）
    """
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int,
                 d_ff: int = None, max_len: int = 1024,
                 attn_dropout: float = 0.0, resid_dropout: float = 0.0, emb_dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        d_ff = d_ff or (4 * d_model)  # 常见设置：FFN 维度为 4*d_model

        # 1) 词嵌入与位置嵌入（可学习）
        self.tok_emb = nn.Embedding(vocab_size, d_model)         # [B,L] -> [B,L,d_model]
        self.pos_emb = nn.Embedding(max_len, d_model)            # [L]   -> [L,d_model]
        self.emb_dropout = nn.Dropout(emb_dropout)

        # 2) 堆叠 N 层 GPT Block
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_head, d_ff, attn_dropout=attn_dropout, resid_dropout=resid_dropout)
            for _ in range(n_layer)
        ])

        # 3) 最后再做一次 LayerNorm（GPT-2 风格）
        self.ln_f = nn.LayerNorm(d_model)

        # 4) 语言建模头：映射到词表大小（通常与 tok_emb 权重共享）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享：把输出头权重与输入词嵌入权重绑到一起（提升收敛/减少参数）
        self.lm_head.weight = self.tok_emb.weight

        # 参数初始化（与 GPT-2 类似的简化版）
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 线性层：正态初始化
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # 嵌入层：正态初始化
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, need_attn: bool = False):
        """
        idx: [B,L] 的 token id 序列
        返回：
          logits: [B,L,vocab_size] 词概率未归一化分数
          attn_list(可选): 每层第0个头的注意力权重（便于观察），形如 List[[B,h,L,L], ...]
        """
        B, L = idx.shape
        assert L <= self.max_len, "输入长度超过了 max_len"

        # 1) 词嵌入 + 位置嵌入
        tok = self.tok_emb(idx)                                 # [B,L,d_model]
        pos_ids = torch.arange(L, device=idx.device)            # [L]
        pos = self.pos_emb(pos_ids).unsqueeze(0)                # [1,L,d_model]
        x = tok + pos                                           # 位置通过相加注入
        x = self.emb_dropout(x)

        # 2) 构造因果掩码（一次复用到所有层）
        attn_mask = build_causal_mask(L, device=idx.device)     # [1,1,L,L]

        # 3) 经过 N 个 Transformer Block
        attn_list = []
        for block in self.blocks:
            if need_attn:
                x, attn_w = block(x, attn_mask, need_attn=True) # 取本层注意力
                attn_list.append(attn_w)                        # [B,h,L,L]
            else:
                x = block(x, attn_mask)

        # 4) 末端 LayerNorm
        x = self.ln_f(x)                                        # [B,L,d_model]

        # 5) 语言建模头（共享词嵌入权重）
        logits = self.lm_head(x)                                # [B,L,vocab_size]
        if need_attn:
            return logits, attn_list
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None):
        """
        简单的自回归生成（贪心/采样）：
          - temperature 控制随机性；top_k 进行截断采样
        """
        for _ in range(max_new_tokens):
            # 若长度超过窗口，裁切右侧窗口（经济实现）
            idx_cond = idx[:, -self.max_len:]

            # 前向得到最后一步的 logits
            logits = self(idx_cond)                             # [B,L,vocab]
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # 取最后位置

            # 可选 top-k 截断
            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k)
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)

            # 转成概率并采样一个 token（或取 argmax）
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B,1]
            # 拼接到序列末尾
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# =========================
# 最小可运行 Demo
# =========================
if __name__ == "__main__":
    # 超小词表 + 超小模型，便于 CPU 快速跑通
    vocab_size = 100
    d_model = 64
    n_layer = 2
    n_head = 4
    max_len = 32
    model = GPT(vocab_size, d_model, n_layer, n_head, d_ff=4*d_model, max_len=max_len,
                attn_dropout=0.1, resid_dropout=0.1, emb_dropout=0.1)

    # 假数据：两个样本，每条序列长度为 8
    B, L = 2, 8
    x = torch.randint(0, vocab_size, (B, L))

    # 前向：拿到 logits 与（可选）注意力权重
    logits, attn_list = model(x, need_attn=True)
    print("输入形状:", x.shape)                   # [2,8]
    print("logits 形状:", logits.shape)           # [2,8,100]
    print("第1层注意力形状:", attn_list[0].shape)  # [2,4,8,8] -> [B,h,L,L]

    # 生成：从第一条样本开始续写 5 个 token（top-k 采样示例）
    out_ids = model.generate(x[:1], max_new_tokens=5, temperature=1.0, top_k=10)
    print("生成后序列形状:", out_ids.shape)        # [1, 8+5]
    print("生成的 token 序列:", out_ids.tolist())
