# =========================
# 通用版：模块封装（Dot 与 RBF 两种“相似度”）
# =========================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(precision=4, sci_mode=False)

class ScaledDotProductAttention(nn.Module):
    """
    标准缩放点积注意力：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    输入形状：
      Q: [B, L_q, d_k], K: [B, L_k, d_k], V: [B, L_k, d_v]
    输出：
      O: [B, L_q, d_v], attn: [B, L_q, L_k], scores: [B, L_q, L_k]
    """
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        # 计算打分：QK^T / sqrt(d_k)
        d_k = Q.size(-1)                                              # 取特征维度 d_k
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)           # [B, L_q, L_k]

        # 可选掩码（如填充位置、未来时刻），被mask的地方置为 -inf，softmax 后趋近 0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax 归一化，得到注意力权重
        attn = F.softmax(scores, dim=-1)                               # [B, L_q, L_k]
        # 对 V 做加权求和，得到输出
        out = attn @ V                                                 # [B, L_q, d_v]
        return out, attn, scores


class RBFDistanceAttention(nn.Module):
    """
    基于“距离/RBF”的注意力：weights ~ softmax( - ||q-k||^2 / (2σ^2) )
    与高斯核相同；σ 为带宽超参（可设为 sqrt(d_k) 或学习参数）
    """
    def __init__(self, sigma=None, learnable_sigma=False):
        super().__init__()
        self.sigma = sigma
        self.learnable_sigma = learnable_sigma
        if learnable_sigma:
            # 若需要学习 σ，用对数参数避免 σ 为负
            init_sigma = 1.0 if sigma is None else float(sigma)
            self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma)))
        else:
            self.log_sigma = None

    def forward(self, Q, K, V, mask=None):
        # Q/K: [B, L, d]
        B, Lq, d = Q.shape
        _, Lk, _ = K.shape

        # σ（带宽）确定：优先用可学习参数，否则用传入值，再否则用 sqrt(d)
        if self.learnable_sigma:
            sigma = torch.exp(self.log_sigma)                          # 确保 σ > 0
        else:
            sigma = math.sqrt(d) if self.sigma is None else float(self.sigma)

        # 计算成对平方距离：dist^2 = ||q||^2 + ||k||^2 - 2 q·k
        q_norm2 = (Q ** 2).sum(dim=-1, keepdim=True)                   # [B, L_q, 1]
        k_norm2 = (K ** 2).sum(dim=-1).unsqueeze(-2)                   # [B, 1, L_k]
        dot_qk  = Q @ K.transpose(-2, -1)                              # [B, L_q, L_k]
        dist2   = q_norm2 + k_norm2 - 2.0 * dot_qk                     # [B, L_q, L_k]

        # RBF 打分：-dist^2 / (2σ^2)
        scores = - dist2 / (2.0 * sigma * sigma)

        # 掩码处理（同上）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax 得权重
        attn = F.softmax(scores, dim=-1)                               # [B, L_q, L_k]
        # 加权求和得到输出
        out = attn @ V                                                 # [B, L_q, d_v]
        return out, attn, scores, dist2


# ============== 演示：随机输入（可自由调整 B/L/d） ==============
B, L, d = 1, 4, 4                         # 1个样本，4个token，特征维度4
x = torch.tensor([[  # 手写一个稳定可复现的小张量，便于观察
    [1.0,  0.5,  0.0, -0.5],
    [0.2, -0.1,  0.8,  0.3],
    [0.0,  1.0,  0.5,  0.0],
    [0.9,  0.1, -0.3,  0.2],
]], dtype=torch.float32)                  # [1, 4, 4]

# 用最朴素的线性层把 x 投影出 Q/K/V（真实模型里这三个是不同矩阵）
W_Q = nn.Linear(d, d, bias=False)
W_K = nn.Linear(d, d, bias=False)
W_V = nn.Linear(d, d, bias=False)

# 为了可重复，我们把权重设成单位矩阵（等价 Q=K=V=x；实际训练中是可学习参数）
with torch.no_grad():
    W_Q.weight.copy_(torch.eye(d))
    W_K.weight.copy_(torch.eye(d))
    W_V.weight.copy_(torch.eye(d))

Q = W_Q(x)   # [1, 4, 4]
K = W_K(x)   # [1, 4, 4]
V = W_V(x)   # [1, 4, 4]

dot_attn = ScaledDotProductAttention()
rbf_attn = RBFDistanceAttention(sigma=math.sqrt(d))  # σ=√d，与点积缩放量纲匹配

out_dot, attn_dot, scores_dot = dot_attn(Q, K, V)
out_rbf, attn_rbf, scores_rbf, dist2 = rbf_attn(Q, K, V)

print("=== 随机输入：缩放点积注意力 ===")
print("scores(QK^T/√d) =\n", scores_dot[0])
print("weights =\n", attn_dot[0])
print("out =\n", out_dot[0])

print("\n=== 随机输入：RBF/距离注意力 ===")
print("dist^2 =\n", dist2[0])
print("scores(-dist^2/(2σ^2)) =\n", scores_rbf[0])
print("weights =\n", attn_rbf[0])
print("out =\n", out_rbf[0])

# 小结提示：若向量范数相近（或做过归一化），两种注意力的权重模式会相近；
# 差异主要由 σ（带宽）、向量范数分布与缩放因子决定。
