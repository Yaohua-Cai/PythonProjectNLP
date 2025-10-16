# =========================
# 自注意力：2 词、2 维的小例子
# =========================
import math
import torch

torch.set_printoptions(precision=4, sci_mode=False)  # 设置打印格式，保留4位小数，便于肉眼比对

# --------- 构造一个最简单的场景（2个token，每个向量2维）---------
# 说明：这里我们人为指定 Q、K、V，便于演算与对照；实际训练中 Q/K/V 都是线性层学出来的
Q = torch.tensor([[1.0, 0.0],   # token1 的查询向量
                  [0.0, 1.0]])  # token2 的查询向量
K = torch.tensor([[1.0, 0.0],   # token1 的键向量
                  [0.0, 1.0]])  # token2 的键向量
V = torch.tensor([[10.0, 0.0],  # token1 的值向量（信息载体）
                  [0.0, 10.0]]) # token2 的值向量

# 为了符合自注意力的批次与序列维度约定，添加 batch 维（1）与序列维（2）
Q = Q.unsqueeze(0)  # [1, 2, 2]
K = K.unsqueeze(0)  # [1, 2, 2]
V = V.unsqueeze(0)  # [1, 2, 2]

# --------- 标准缩放点积注意力（Scaled Dot-Product Attention）---------
d_k = Q.size(-1)                         # d_k 是 K/Q 的最后一维（特征维）
scores_dot = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)  # [1, 2, 2]，公式：QK^T / sqrt(d_k)
weights_dot = torch.softmax(scores_dot, dim=-1)          # 对每行做 softmax，得到注意力权重
out_dot = weights_dot @ V                                 # 输出：权重对 V 的加权求和

print("=== 缩放点积注意力（Dot-Product） ===")
print("QK^T/sqrt(d_k) =\n", scores_dot)
print("softmax 权重 =\n", weights_dot)
print("输出 O =\n", out_dot)

# --------- 基于“距离/RBF”的注意力（Gaussian / RBF kernel）---------
# 思想：相似度 ~ exp( - ||q - k||^2 / (2σ^2) )，距离越小，相似度越大
# 计算平方距离：||q-k||^2 = ||q||^2 + ||k||^2 - 2 q·k
q_norm2 = (Q ** 2).sum(dim=-1, keepdim=True)             # [1, 2, 1]，每个 q 的平方范数
k_norm2 = (K ** 2).sum(dim=-1).unsqueeze(-2)             # [1, 1, 2]，每个 k 的平方范数（扩展到行）
dot_qk  = Q @ K.transpose(-2, -1)                        # [1, 2, 2]，点积项
dist2   = q_norm2 + k_norm2 - 2.0 * dot_qk               # [1, 2, 2]，成对的平方欧氏距离

# 选择一个 σ（带宽）。常见做法之一：σ 取 sqrt(d_k)，可与“缩放点积”的量纲对应
sigma = math.sqrt(d_k)

# RBF“打分”= -dist2 / (2σ^2)，然后再 softmax 正规化
scores_rbf = - dist2 / (2.0 * sigma * sigma)             # [1, 2, 2]
weights_rbf = torch.softmax(scores_rbf, dim=-1)          # [1, 2, 2]
out_rbf = weights_rbf @ V                                 # [1, 2, 2]

print("\n=== 距离注意力（RBF/Gaussian） ===")
print("平方距离 dist^2 =\n", dist2)
print("RBF 打分 = -dist^2/(2σ^2) =\n", scores_rbf)
print("softmax 权重 =\n", weights_rbf)
print("输出 O =\n", out_rbf)

# --------- 小结：为什么“点积注意力”和“距离注意力”会产生相似的偏好？---------
# 恒等式： 2 q·k = ||q||^2 + ||k||^2 - ||q - k||^2
# 若 ||q|| 与 ||k|| 的范数相近（或归一化），则“点积大” ≈ “距离小”，
# 因而二者的 softmax 权重趋势类似（具体受 σ 与缩放因子影响）。
