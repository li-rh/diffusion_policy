import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # 获取输入张量所在的设备(CPU/GPU)
        device = x.device
        # 计算位置编码维度的一半
        half_dim = self.dim // 2
        # 计算位置编码的缩放因子(基于10000的对数缩放)
        emb = math.log(10000) / (half_dim - 1)
        # 生成位置编码的指数衰减序列
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 将输入x与位置编码序列进行外积运算
        emb = x[:, None] * emb[None, :]
        # 拼接正弦和余弦部分作为最终的位置编码
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

if __name__ == "__main__":
    # 创建一个SinusoidalPosEmb实例，维度为10
    pos_emb = SinusoidalPosEmb(10)
    # 生成一个形状为(5, 1)的随机输入张量
    x = torch.randn(5, 1)
    # 计算输入张量的位置编码
    y = pos_emb(x)
    # 打印输入张量和位置编码的形状
    print(x.shape, y.shape)