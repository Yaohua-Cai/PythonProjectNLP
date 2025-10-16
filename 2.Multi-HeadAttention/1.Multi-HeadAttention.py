# ==========================================
# 多头注意力（点积 vs 距离/RBF）——从零实现
# ==========================================
import math                           # 数学函数（如开平方）
import torch                          # PyTorch 主库
import torch.nn as nn                 # 神经网络模块
import torch.nn.functional as F       # 常用函数（softmax 等）

torch.manual_seed(0)                  # 固定随机种子，保证可复现
torch.set_printoptions(precision=4, sci_mode=False)  # 打印小数更友好


class MultiHeadAttentionDistance(nn.Module):
    """
    一个可切换相似度的多头注意力：
    - use_rbf=False 时：标准“缩放点积注意力”（Scaled Dot-Product Attention）
    - use_rbf=True  时：基于距离的注意力（高斯/RBF核思想）
    """
    def __init__(self, d_model: int, num_heads: int,
                 use_rbf: bool = False,
                 sigma: float | None = None,
                 learnable_sigma: bool = False):
        """
        d_model: 词向量总维度（会被均分到各个头）
        num_heads: 注意力头的个数
        use_rbf: 是否使用 RBF（距离）相似度；False 则使用标准点积
        sigma: RBF 的带宽，若不指定默认用 sqrt(d_head)
        learnable_sigma: 是否让 sigma 可学习（作为参数训练）
        """
        super().__init__()                        # 调用父类构造
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"  # 保证均分

        self.d_model = d_model                    # 保存输入总维度
        self.num_heads = num_heads                # 保存头数
        self.d_head = d_model // num_heads        # 每个头的维度 = 总维度 / 头数
        self.use_rbf = use_rbf                    # 标记是否用 RBF 距离注意力

        # 三个线性映射：把输入映射到 Q/K/V 空间（每个都是 d_model -> d_model）
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 头拼接后的输出映射（d_model -> d_model）
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # RBF 带宽 sigma 的处理：可固定，也可设成可学习参数
        if learnable_sigma:
            # 用 log_sigma 作为可学习参数，指数映射保证 sigma>0
            init_sigma = (self.d_head ** 0.5) if sigma is None else float(sigma)
            self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma), dtype=torch.float32))
        else:
            self.log_sigma = None                  # 不学习时置空
            # 固定值：若不传入，则默认 sqrt(d_head)
            self.sigma = (self.d_head ** 0.5) if sigma is None else float(sigma)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        把 [B, L, d_model] 拆成 [B, h, L, d_head]
        B: batch size, L: 序列长度, h: 头数, d_head: 每头维度
        """
        B, L, _ = x.shape                          # 取出 batch 和序列长度
        # 先 reshape 成 [B, L, h, d_head]，再把头维挪到前面 -> [B, h, L, d_head]
        return x.view(B, L, self.num_heads, self.d_head).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        把 [B, h, L, d_head] 合并回 [B, L, d_model]
        """
        B, h, L, d_head = x.shape                  # 拆出各维度
        # 先把头维放回到后面 [B, L, h, d_head]，再 view 合并为 [B, L, d_model]
        return x.permute(0, 2, 1, 3).contiguous().view(B, L, h * d_head)

    def _scores_dot(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        标准“缩放点积”打分：scores = Q @ K^T / sqrt(d_head)
        Q/K 形状均为 [B, h, L, d_head]；返回 [B, h, L, L]
        """
        # K 转置最后两维，从 [B,h,L,d] -> [B,h,d,L]，以便做批量矩阵乘
        KT = K.transpose(-2, -1)
        # 矩阵乘得到原始打分 [B,h,L,L]
        scores = Q @ KT
        # 按论文缩放，避免数值过大导致 softmax 太“陡”
        scores = scores / math.sqrt(self.d_head)
        return scores

    def _scores_rbf(self, Q: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        基于距离（RBF/高斯核）打分：
        scores = - ||q - k||^2 / (2 * sigma^2)
        返回 (scores, dist2)，方便外面可视化距离矩阵
        """
        # 计算每个 query 的平方范数，形状 [B,h,L,1]
        q_norm2 = (Q ** 2).sum(dim=-1, keepdim=True)
        # 计算每个 key 的平方范数，形状 [B,h,1,L]
        k_norm2 = (K ** 2).sum(dim=-1).unsqueeze(-2)
        # 计算批量点积 [B,h,L,L]，用于展开 ||q-k||^2 = ||q||^2 + ||k||^2 - 2 q·k
        dot_qk = Q @ K.transpose(-2, -1)
        # 成对平方距离 dist^2，形状 [B,h,L,L]
        dist2 = q_norm2 + k_norm2 - 2.0 * dot_qk

        # 取得 sigma：可学习则取 exp(log_sigma)，否则用固定值
        if self.log_sigma is not None:
            sigma = torch.exp(self.log_sigma)      # 标量张量，保证 >0
        else:
            sigma = torch.tensor(self.sigma, dtype=Q.dtype, device=Q.device)

        # RBF 打分：距离越小分数越大（负号）
        scores = - dist2 / (2.0 * sigma * sigma)
        return scores, dist2

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                return_intermediates: bool = False):
        """
        x: [B, L, d_model] 的输入序列
        mask: 可选的注意力掩码 [B, 1, L, L]（1 表示可见，0 表示屏蔽）
        return_intermediates: 是否返回中间矩阵（scores、dist2 等），便于教学打印
        """
        # 计算 Q、K、V 的线性投影；形状均为 [B, L, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 拆成多头形状 [B, h, L, d_head]
        Qh = self._split_heads(Q)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)

        # 根据相似度类型计算打分矩阵 scores：[B, h, L, L]
        if not self.use_rbf:
            scores = self._scores_dot(Qh, Kh)      # 点积打分
            dist2 = None                            # 点积模式下无距离矩阵
        else:
            scores, dist2 = self._scores_rbf(Qh, Kh)  # RBF 距离打分，同时拿到 dist^2

        # 如提供 mask（如解码器的“因果掩码”或 padding 掩码），则把不可见位置置为 -inf
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 对最后一维（每个 query 对所有 key）做 softmax，得到注意力权重 α
        attn = F.softmax(scores, dim=-1)           # [B, h, L, L]

        # 用注意力权重对 V 做加权求和，得到每个头的输出
        Oh = attn @ Vh                              # [B, h, L, d_head]

        # 合并多头输出回到 [B, L, d_model]
        O = self._merge_heads(Oh)

        # 最后做一次输出线性变换（如论文中的 W_o）
        O = self.W_o(O)

        # 是否需要把中间变量（scores、attn、dist2）返回出去，便于教学打印
        if return_intermediates:
            return O, attn, scores, dist2
        else:
            return O


# ============================
# 下面是一个可运行的小 Demo
# ============================
if __name__ == "__main__":
    # ------- 构造一个玩具输入：B=1, L=4, d_model=8 -------
    B, L, d_model = 1, 4, 8            # 1 个样本，4 个 token，8 维表示
    num_heads = 2                      # 2 个注意力头（每头维度 d_head=4）

    # 手工构造一个 4x8 的序列，让第 1/4 个 token 更接近（便于观察“近的更相关”）
    x = torch.tensor([[
        [ 1.0,  0.8,  0.1,  0.0,   0.2, -0.1,  0.0,  0.0],  # token 0
        [ 0.2, -0.5,  0.9,  0.3,   0.1,  0.2, -0.4,  0.2],  # token 1
        [ 0.0,  1.0,  0.6,  0.1,  -0.1,  0.0,  0.3,  0.2],  # token 2
        [ 1.1,  0.7,  0.0, -0.1,   0.2, -0.1, -0.1,  0.1],  # token 3（与 token0 相似）
    ]], dtype=torch.float32)                                # 形状 [1,4,8]

    # ------- 实例化两种注意力：点积版 与 RBF（距离）版 -------
    mha_dot = MultiHeadAttentionDistance(d_model=d_model, num_heads=num_heads,
                                         use_rbf=False)     # 点积注意力
    mha_rbf = MultiHeadAttentionDistance(d_model=d_model, num_heads=num_heads,
                                         use_rbf=True,      # 距离注意力
                                         sigma=None,        # 不指定则默认 sqrt(d_head)
                                         learnable_sigma=False)

    # ------- 为了对比公平：把两者的线性层权重都设成“单位矩阵” -------
    # 这样 Q=K=V≈输入 x，本例更容易用“距离直觉”理解注意力权重
    with torch.no_grad():
        eye = torch.eye(d_model)                            # 8x8 单位阵
        for m in [mha_dot, mha_rbf]:
            m.W_q.weight.copy_(eye)                         # W_q = I
            m.W_k.weight.copy_(eye)                         # W_k = I
            m.W_v.weight.copy_(eye)                         # W_v = I
            m.W_o.weight.copy_(eye)                         # W_o = I

    # ------- 前向：返回中间矩阵，便于打印 -------
    out_dot, attn_dot, scores_dot, _      = mha_dot(x, return_intermediates=True)
    out_rbf, attn_rbf, scores_rbf, dist2  = mha_rbf(x, return_intermediates=True)

    # ------- 打印形状核对 -------
    print("输入 x 形状:", x.shape)                          # 期望 [1,4,8]
    print("输出（点积）形状:", out_dot.shape)                # 期望 [1,4,8]
    print("输出（RBF）形状:",  out_rbf.shape)                # 期望 [1,4,8] 与上相同

    # ------- 辅助函数：把每个头的矩阵漂亮地打印出来 -------
    def print_head_matrices(title: str, scores: torch.Tensor, attn: torch.Tensor,
                            dist2: torch.Tensor | None = None):
        """
        scores: [B,h,L,L] 的打分矩阵（点积或 RBF 打分）
        attn:   [B,h,L,L] 的 softmax 权重矩阵
        dist2:  [B,h,L,L] 的成对平方距离（仅 RBF 有）
        """
        print(f"\n==== {title} ====")
        B, h, L, _ = attn.shape
        for head in range(h):
            print(f"\n-- 头 {head} --")
            if dist2 is not None:
                print("dist^2 矩阵：\n", dist2[0, head])
            print("scores 矩阵：\n", scores[0, head])
            print("softmax 注意力权重：\n", attn[0, head])

    # ------- 分别打印 点积 与 RBF 的每头权重、打分与（RBF 的）距离 -------
    print_head_matrices("点积注意力（Scaled Dot-Product）", scores_dot, attn_dot, dist2=None)
    print_head_matrices("RBF 距离注意力（Gaussian Kernel）", scores_rbf, attn_rbf, dist2=dist2)

    # ------- 小结：观察差异 -------
    # 由于本例 Q=K=V≈x，且 RBF 的 sigma≈sqrt(d_head)，
    # 往往可以看到两种注意力在“更相近的 token 上分配更高权重”的趋势是一致的。
    # 差异来自：缩放系数、向量范数分布、以及 RBF 带宽 sigma 的取值。
