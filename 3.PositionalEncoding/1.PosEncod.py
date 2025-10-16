# =========================
# 正弦-余弦位置编码（可泛化版本）
# =========================
import math                          # 导入数学库，用于开方/对数等运算
import torch                         # 导入 PyTorch 主库
import torch.nn as nn                # 导入神经网络模块（nn.Module、nn.Embedding 等）
import torch.nn.functional as F      # 常用函数（这里主要用不到，保留以便扩展）

class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦-余弦位置编码：
    PE(pos, 2i)   = sin( pos / 10000^{2i/d_model} )
    PE(pos, 2i+1) = cos( pos / 10000^{2i/d_model} )
    其中 pos 为位置索引（0..max_len-1），i 为维度索引。
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        # 调用父类构造函数，完成 nn.Module 初始化
        super().__init__()
        # 保存模型维度与最大序列长度，方便调试/打印
        self.d_model = d_model
        self.max_len = max_len
        # Dropout 用于训练时对位置编码后的表示做正则（可选）
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 创建位置编码表 pe，形状 [max_len, d_model]，先用 0 填充
        pe = torch.zeros(max_len, d_model)              # 张量大小为 (max_len, d_model)
        # 生成位置向量 pos = [0, 1, 2, ..., max_len-1]，并在列维度上扩展为 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # 生成频率 div_term，针对偶数维（步长为 2）构建不同的缩放频率
        # 公式中的 10000^{2i/d_model} 等价于 exp( (2i/d_model) * ln(10000) )
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        # 对偶数维（索引 0,2,4,...) 赋值 sin(position * div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 对奇数维（索引 1,3,5,...) 赋值 cos(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 扩展 batch 维度，得到 [1, max_len, d_model]，便于与输入 [B, L, d_model] 相加
        pe = pe.unsqueeze(0)
        # register_buffer 会把 pe 当作“模型状态”存起来（参与 .to(device) / 保存），但不当作可训练参数
        self.register_buffer("pe", pe)                  # self.pe 的形状为 [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
          x: [B, L, d_model] 的 token 表示（一般来自词嵌入或上一层输出）
        返回:
          x_pos: [B, L, d_model]，在 x 上加上对应位置的编码后，再过 dropout
        """
        # 从缓冲区 self.pe 中切出当前序列长度 L 对应的位置编码，形状 [1, L, d_model]
        pe_slice = self.pe[:, : x.size(1), :]
        # 将位置编码与输入逐元素相加（广播到 B 维），得到带位置信息的表示
        x = x + pe_slice
        # 训练时可加 Dropout，推理时为恒等映射
        return self.dropout(x)


# ============== 一个最小可运行示例（演示 “怎么加到嵌入上”） ==============
if __name__ == "__main__":
    torch.manual_seed(0)                            # 固定随机种子，保证可复现
    B, L, d_model = 2, 8, 16                        # B=2 个样本，序列长度 L=8，通道维 d_model=16
    vocab_size = 1000                               # 假设词表大小 1000

    # 构造词嵌入层：把 token id 映射到 d_model 维的向量空间
    tok_emb = nn.Embedding(vocab_size, d_model)     # nn.Embedding 会随机初始化参数，需要训练
    # 构造位置编码模块：最大支持序列 512 长度，不用 dropout
    pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=512, dropout=0.0)

    # 构造两个样本的“假 token 序列”，形状 [B, L]，元素范围 [0, vocab_size)
    x_ids = torch.randint(low=0, high=vocab_size, size=(B, L))
    # 通过嵌入层得到稠密表示，形状 [B, L, d_model]
    x = tok_emb(x_ids)
    # 将位置编码加到词向量上，得到具有“顺序信息”的表达
    x_with_pos = pos_enc(x)

    # 打印形状确认
    print("token ids 形状:", x_ids.shape)           # 期望 [2, 8]
    print("词嵌入形状:", x.shape)                   # 期望 [2, 8, 16]
    print("加位置编码后形状:", x_with_pos.shape)     # 期望 [2, 8, 16]

    # 演示：查看前 4 个位置、前 6 个维度的“纯位置编码数值”（不含词向量）
    # 注意：pos_enc.pe 的形状为 [1, max_len, d_model]
    print("位置编码示例（前 4 个位置、前 6 维）：\n", pos_enc.pe[0, :4, :6])

    # 验证：同一位置编码向量的数值范围在 [-1, 1]，且不同位置会有不同“波形签名”
