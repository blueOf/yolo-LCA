from ultralytics import YOLO
from ultralytics.nn.modules import Conv
import torch.nn as nn
import torch.nn.functional as F
import torch
from thop import profile


class FastLocalDETRBlock(nn.Module):
    def __init__(self, c, cout, sight=2, head_num=4, N=1):
        super().__init__()
        self.c, self.cout, self.sight, self.head_num, self.N = (
            c,
            cout,
            sight,
            head_num,
            N,
        )
        self.kernel_size = 2 * sight + 1
        self.input_norm = nn.LayerNorm(c)
        self.proj = nn.Linear(c, cout) if c != cout else nn.Identity()

        # 核心点：用 1x1 卷积代替 Linear，并放在最前面
        # 这样 unfold 的就是投影后的结果，减少后续计算
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=sight)

        # 编码器：N 层邻域自注意力，每层输出作为下一层的输入
        self.encoder_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "qkv_conv": nn.Conv2d(c, 3 * c, 1),  # 一次性完成邻居的 QKV
                        "norm_sa": nn.LayerNorm(c),
                    }
                )
                for _ in range(N)
            ]
        )

        # 解码器：N 层交叉注意力，使用原始 center_x 作为初始 Q
        self.decoder_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "q_center": nn.Linear(c, c),
                        "norm_ca": nn.LayerNorm(c),
                        "ffn": nn.Sequential(
                            nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c)
                        ),
                        "norm_ffn": nn.LayerNorm(c),
                    }
                )
                for _ in range(N)
            ]
        )

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        Np = self.kernel_size**2
        d = C // self.head_num

        x = (
            self.input_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        )  # (B, C, H, W)

        # 保存原始 center_x 作为解码器的初始 Q
        original_center_x = x.flatten(2).transpose(1, 2)  # (B, L, C)

        # --- 编码阶段：N 层，每层输出作为下一层的 Q/K/V ---
        cur_x = x
        center_idx = (Np - 1) // 2
        for layer in self.encoder_layers:
            qkv = layer["qkv_conv"](cur_x)  # (B, 3C, H, W)
            # 单次 permute+contiguous 完成所有头分割，减少内存复制次数
            qkv_heads = (
                self.unfold(qkv)                       # (B, 3C*Np, L)
                .view(B, 3, self.head_num, d, Np, L)
                .permute(1, 0, 5, 2, 4, 3)             # (3, B, L, head, Np, d)
                .contiguous()
                .view(3, B * L, self.head_num, Np, d)
            )
            q_nb, k_nb, v_nb = qkv_heads.unbind(0)    # 每个 (B*L, head, Np, d)，连续内存

            # 手动多头注意力：Flash Attention 要求序列长度对齐（64/128），Np=25 不满足，
            # 且 B*L 极大时会超出 CUDA 内核网格限制，故保留显式算子；
            # 一步融合避免保留中间注意力矩阵
            attn_out = (
                (q_nb @ k_nb.transpose(-2, -1)) * (d ** -0.5)
            ).softmax(dim=-1) @ v_nb                               # (B*L, head, Np, d)
            nb_encoded = layer["norm_sa"](
                attn_out.transpose(1, 2).reshape(B * L, Np, C)
            )  # (B*L, Np, C)

            # 取中心位置特征更新空间特征图，作为下一层编码器的输入
            cur_x = (
                nb_encoded[:, center_idx]   # (B*L, C)
                .view(B, L, C)
                .permute(0, 2, 1)           # (B, C, L)
                .contiguous()
                .view(B, C, H, W)
            )

        # 预计算解码器所需的 K/V（头分割格式），避免循环内重复布局转换
        nb_encoded_kv = (
            nb_encoded
            .view(B * L, Np, self.head_num, d)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # (B*L, head, Np, d)

        # --- 解码阶段：N 层，使用原始 center_x 作为 Q，最终编码输出作为 K/V ---
        center_x = original_center_x
        for layer in self.decoder_layers:
            # 直接 view 到头分割格式，省去一次 transpose；
            # 原始实现以 C**-0.5 (= (head_num*d)**-0.5) 为缩放因子（非标准 d**-0.5），此处保持一致
            qc = layer["q_center"](center_x).view(B * L, self.head_num, 1, d)
            attn_c = (qc @ nb_encoded_kv.transpose(-2, -1)) * (C ** -0.5)  # (B*L, head, 1, Np)
            ca_out = (
                (attn_c.softmax(dim=-1) @ nb_encoded_kv)  # (B*L, head, 1, d)
                .transpose(1, 2)
                .reshape(B, L, C)
            )  # (B, L, C)

            center_x = layer["norm_ca"](center_x + ca_out)
            center_x = layer["norm_ffn"](center_x + layer["ffn"](center_x))

        return self.proj(center_x).transpose(1, 2).view(B, self.cout, H, W)


class ConvWithAttention(nn.Module):
    """
    Implementation of a module that performs convolution to generate K and V,
    then uses the original input as Q to perform dot-product attention.
    - N: Number of repeated layers
    - size: Kernel size for the convolution
    - Maintains the same spatial dimensions for KV.
    """

    def __init__(self, c1, c2, n=1, size=3, head_num=4):
        super().__init__()
        self.c1 = c1
        self.c2 = (
            c2  # Reserved for YOLO architecture compatibility, but output stays c1
        )
        self.n = n
        self.size = size
        self.padding = size // 2
        self.head_num = head_num
        self.head_dim = c1 // head_num
        assert (
            c1 % head_num == 0
        ), f"c1 ({c1}) must be divisible by head_num ({head_num})"

        self.norm_q = nn.InstanceNorm2d(c1, affine=True)

        # Encoder: Convolutional layers to generate context (K, V)
        self.encoder = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "conv": Conv(c1, c1, size, p=self.padding),
                    }
                )
                for _ in range(n)
            ]
        )

        # Decoder Projections
        self.decoder_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "q_proj": Conv(c1, c1, 1, act=False),
                        "kv_proj": Conv(c1, c1, 1, act=False),
                    }
                )
                for _ in range(n)
            ]
        )

    def forward(self, x):
        """
        Forward pass for ConvWithAttention.
        x: (B, c1, H, W)
        """
        B, C, H, W = x.shape
        L = H * W

        # 1. Encoder processing (Generating context K, V)
        # We process the entire encoder first to get the final context
        context = self.norm_q(x)
        for layer in self.encoder:
            context = layer["conv"](context)  # (B, C, H, W)

        # 2. Decoder processing (N repeated steps like Transformer)
        # Using the original normalized x as the initial query-side input
        res = x
        for layer in self.decoder_layers:
            # Normalize and Project Q
            q = layer["q_proj"](self.norm_q(res))  # (B, C, H, W)

            # Project KV from the encoder's final context
            kv = layer["kv_proj"](context)  # (B, C, H, W)
            k = v = kv

            # 3. Optimized Property-wise Attention (Avoids L x L matrix)
            # Based on user requirement: "each c only interacts with its corresponding vector"
            # This implements a localized channel-wise scaling/attention

            # Compute attention score per pixel and per channel
            # Scaling factors: (B, C, H, W)
            attn_weight = (q * k) * (self.head_dim**-0.5)
            attn_weight = torch.softmax(
                attn_weight, dim=1
            )  # Across channels or just sigmoid

            out = attn_weight * v

            # Residual connection
            res = res + out

        return res


class FastLocalDETR_PE(nn.Module):

    def __init__(self, c, cout, sight=2, head_num=4, N=1):
        super().__init__()
        self.c, self.cout, self.sight, self.head_num, self.N = (
            c,
            cout,
            sight,
            head_num,
            N,
        )
        self.kernel_size = 2 * sight + 1
        self.input_norm = nn.LayerNorm(c)
        self.proj = nn.Linear(c, cout) if c != cout else nn.Identity()

        # 核心点：用 1x1 卷积代替 Linear，并放在最前面
        # 这样 unfold 的就是投影后的结果，减少后续计算
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=sight)

        # 编码器：N 层邻域自注意力，每层输出作为下一层的输入
        self.encoder_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "qkv_conv": nn.Conv2d(c, 3 * c, 1),  # 一次性完成邻居的 QKV
                        "norm_sa": nn.LayerNorm(c),
                    }
                )
                for _ in range(N)
            ]
        )

        # 解码器：N 层交叉注意力，使用原始 center_x 作为初始 Q
        self.decoder_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "q_center": nn.Linear(c, c),
                        "norm_ca": nn.LayerNorm(c),
                        "ffn": nn.Sequential(
                            nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c)
                        ),
                        "norm_ffn": nn.LayerNorm(c),
                    }
                )
                for _ in range(N)
            ]
        )
        self.row_pos_embed = nn.Parameter(torch.randn(self.kernel_size**2, 1) * 0.02)
        self.col_pos_embed = nn.Parameter(torch.randn(1, c) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        Np = self.kernel_size**2
        d = C // self.head_num
        center_idx = (Np - 1) // 2

        x = (
            self.input_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        )  # (B, C, H, W)

        pos_embed = self.row_pos_embed + self.col_pos_embed  # (Np, C)

        # 保存原始 center_x（含中心位置编码）作为解码器的初始 Q
        original_center_x = x.flatten(2).transpose(1, 2) + pos_embed[center_idx]  # (B, L, C)

        # 预计算位置偏置的广播形状 (1, 1, C, Np, 1)，供编码器首层注入
        pos_bias = pos_embed.t().unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, C, Np, 1)

        # --- 编码阶段：N 层，每层输出作为下一层的 Q/K/V ---
        cur_x = x
        for i, layer in enumerate(self.encoder_layers):
            qkv = layer["qkv_conv"](cur_x)  # (B, 3C, H, W)
            # 在头分割前注入位置编码（仅首层），避免额外布局转换开销
            unfolded_3C = self.unfold(qkv).view(B, 3, C, Np, L)  # 零复制 view
            if i == 0:
                unfolded_3C = unfolded_3C + pos_bias  # 广播加法，结果连续
            # 单次 permute+contiguous 完成所有头分割，减少内存复制次数
            qkv_heads = (
                unfolded_3C
                .view(B, 3, self.head_num, d, Np, L)
                .permute(1, 0, 5, 2, 4, 3)             # (3, B, L, head, Np, d)
                .contiguous()
                .view(3, B * L, self.head_num, Np, d)
            )
            q_nb, k_nb, v_nb = qkv_heads.unbind(0)    # 每个 (B*L, head, Np, d)，连续内存

            # 手动多头注意力：Flash Attention 要求序列长度对齐（64/128），Np=25 不满足，
            # 且 B*L 极大时会超出 CUDA 内核网格限制，故保留显式算子；
            # 一步融合避免保留中间注意力矩阵
            attn_out = (
                (q_nb @ k_nb.transpose(-2, -1)) * (d ** -0.5)
            ).softmax(dim=-1) @ v_nb                               # (B*L, head, Np, d)
            nb_encoded = layer["norm_sa"](
                attn_out.transpose(1, 2).reshape(B * L, Np, C)
            )  # (B*L, Np, C)

            # 取中心位置特征更新空间特征图，作为下一层编码器的输入
            cur_x = (
                nb_encoded[:, center_idx]   # (B*L, C)
                .view(B, L, C)
                .permute(0, 2, 1)           # (B, C, L)
                .contiguous()
                .view(B, C, H, W)
            )

        # 预计算解码器所需的 K/V（头分割格式），避免循环内重复布局转换
        nb_encoded_kv = (
            nb_encoded
            .view(B * L, Np, self.head_num, d)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # (B*L, head, Np, d)

        # --- 解码阶段：N 层，使用原始 center_x 作为 Q，最终编码输出作为 K/V ---
        center_x = original_center_x
        for layer in self.decoder_layers:
            # 直接 view 到头分割格式，省去一次 transpose；
            # 原始实现以 C**-0.5 (= (head_num*d)**-0.5) 为缩放因子（非标准 d**-0.5），此处保持一致
            qc = layer["q_center"](center_x).view(B * L, self.head_num, 1, d)
            attn_c = (qc @ nb_encoded_kv.transpose(-2, -1)) * (C ** -0.5)  # (B*L, head, 1, Np)
            ca_out = (
                (attn_c.softmax(dim=-1) @ nb_encoded_kv)  # (B*L, head, 1, d)
                .transpose(1, 2)
                .reshape(B, L, C)
            )  # (B, L, C)

            center_x = layer["norm_ca"](center_x + ca_out)
            center_x = layer["norm_ffn"](center_x + layer["ffn"](center_x))

        return self.proj(center_x).transpose(1, 2).view(B, self.cout, H, W)


class FastLocalDETR_PE_orin(nn.Module):

    def __init__(self, c, cout, sight=2, head_num=4, N=1):
        super().__init__()
        self.c, self.cout, self.sight, self.head_num, self.N = (
            c,
            cout,
            sight,
            head_num,
            N,
        )
        self.kernel_size = 2 * sight + 1
        self.input_norm = nn.LayerNorm(c)
        self.proj = nn.Linear(c, cout) if c != cout else nn.Identity()

        # 核心点：用 1x1 卷积代替 Linear，并放在最前面
        # 这样 unfold 的就是投影后的结果，减少后续计算
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=sight)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "qkv_conv": nn.Conv2d(c, 3 * c, 1),  # 一次性完成邻居的 QKV
                        "norm_sa": nn.LayerNorm(c),
                        "q_center": nn.Linear(c, c),
                        "norm_ca": nn.LayerNorm(c),
                        "ffn": nn.Sequential(
                            nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c)
                        ),
                        "norm_ffn": nn.LayerNorm(c),
                    }
                )
                for _ in range(N)
            ]
        )
        self.row_pos_embed = nn.Parameter(torch.randn(self.kernel_size**2, 1) * 0.02)
        self.col_pos_embed = nn.Parameter(torch.randn(1, c) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        Np = self.kernel_size**2
        d = C // self.head_num
        center_idx = (Np - 1) // 2

        x = (
            self.input_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        )  # (B, C, H, W)

        pos_embed = self.row_pos_embed + self.col_pos_embed  # (Np, C)
        center_x = x.flatten(2).transpose(1, 2) + pos_embed[center_idx]  # (B, L, C)

        # 预计算位置偏置的广播形状 (1, 1, C, Np, 1)，供编码器首层注入
        pos_bias = pos_embed.t().unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, C, Np, 1)

        for i, layer in enumerate(self.layers):
            qkv = layer["qkv_conv"](x)  # (B, 3C, H, W)
            # 在头分割前注入位置编码（仅首层），避免额外布局转换开销
            unfolded_3C = self.unfold(qkv).view(B, 3, C, Np, L)  # 零复制 view
            if i == 0:
                unfolded_3C = unfolded_3C + pos_bias  # 广播加法，结果连续
            # 单次 permute+contiguous 完成所有头分割，减少内存复制次数
            qkv_heads = (
                unfolded_3C
                .view(B, 3, self.head_num, d, Np, L)
                .permute(1, 0, 5, 2, 4, 3)             # (3, B, L, head, Np, d)
                .contiguous()
                .view(3, B * L, self.head_num, Np, d)
            )
            q_nb, k_nb, v_nb = qkv_heads.unbind(0)    # 每个 (B*L, head, Np, d)，连续内存

            # 手动多头注意力：Flash Attention 要求序列长度对齐（64/128），Np=25 不满足，
            # 且 B*L 极大时会超出 CUDA 内核网格限制，故保留显式算子；
            # 一步融合避免保留中间注意力矩阵
            attn_out = (
                (q_nb @ k_nb.transpose(-2, -1)) * (d ** -0.5)
            ).softmax(dim=-1) @ v_nb                               # (B*L, head, Np, d)
            nb_encoded = layer["norm_sa"](
                attn_out.transpose(1, 2).reshape(B * L, Np, C)
            )  # (B*L, Np, C)

            # --- CA ---
            nb_encoded_kv = (
                nb_encoded
                .view(B * L, Np, self.head_num, d)
                .permute(0, 2, 1, 3)
                .contiguous()
            )  # (B*L, head, Np, d)
            # 直接 view 到头分割格式，省去一次 transpose；
            # 原始实现以 C**-0.5 (= (head_num*d)**-0.5) 为缩放因子（非标准 d**-0.5），此处保持一致
            qc = layer["q_center"](center_x).view(B * L, self.head_num, 1, d)
            attn_c = (qc @ nb_encoded_kv.transpose(-2, -1)) * (C ** -0.5)  # (B*L, head, 1, Np)
            ca_out = (
                (attn_c.softmax(dim=-1) @ nb_encoded_kv)  # (B*L, head, 1, d)
                .transpose(1, 2)
                .reshape(B, L, C)
            )  # (B, L, C)

            center_x = layer["norm_ca"](center_x + ca_out)
            center_x = layer["norm_ffn"](center_x + layer["ffn"](center_x))

            if self.N > 1:
                x = center_x.transpose(1, 2).view(B, C, H, W).contiguous()

        return self.proj(center_x).transpose(1, 2).view(B, self.cout, H, W)


if __name__ == "__main__":
    from thop import profile

    # 设定参数
    C = 64
    H = W = 80
    N = 2
    sight = 2

    # 初始化模块
    block = FastLocalDETR_PE(c=C, cout=C, sight=sight, head_num=4, N=N)
    x = torch.randn((1, C, H, W))

    # 定义自定义算子逻辑以包含矩阵乘法 (SA 和 CA 中的 @ 运算)
    # thop 默认只统计 nn.Module 的参数和计算逻辑
    # 我们通过 profile 运行并打印
    flops, params = profile(block, inputs=(x,))

    print("-" * 30)
    print(f"Input Shape: {x.shape}")
    print(f"Final FLOPs: {flops / 1e9:.3f} G")
    print(f"Final Params: {params / 1e6:.3f} M")

    # 手动计算矩阵运算部分的 FLOPs (thop 可能遗失的部分)
    # L = 80*80 = 6400, Np = 5*5 = 25
    # SA: q(Np, head_dim) @ k(head_dim, Np) -> (Np, Np) 每一行都有一个 L
    # FLOPs_SA = B * L * head * Np * Np * head_dim * 2
    L = H * W
    Np = (2 * sight + 1) ** 2
    head = 4
    head_dim = C // head

    # SA 矩阵乘法约为: B * L * head * Np * Np * head_dim * 2
    # CA 矩阵乘法约为: B * L * head * 1 * Np * head_dim * 2
    # 这里记得乘以 N 层，因为每一层都有这些运算
    sa_matmul_flops = N * (1 * L * head * Np * Np * head_dim * 2)
    ca_matmul_flops = N * (1 * L * head * 1 * Np * head_dim * 2)

    print(
        f"Theoretical SA MatMul FLOPs (Total for {N} layers): {sa_matmul_flops / 1e6:.2f} M"
    )
    print(
        f"Theoretical CA MatMul FLOPs (Total for {N} layers): {ca_matmul_flops / 1e6:.2f} M"
    )
    print("-" * 30)

    out = block(x)
    print(f"Output shape: {out.shape}")
