from ultralytics import YOLO
from ultralytics.nn.modules import Conv
import torch.nn as nn
import torch.nn.functional as F
import torch
from thop import profile
from ultralytics.nn.modules.transformer import LayerNorm2d


class FusionBlockv2(nn.Module):
    def __init__(self, c1, c2, c3):
        """Initialize FusionBlock.

        Args:
            c1 (int): Channels of kv input (first input).
            c2 (int): Channels of q input (second input), also the internal working dimension.
            c3 (int): Output channels.
            head_num (int): Number of attention heads.
        """
        super().__init__()
        # Ensure head_num evenly divides c2
        self.kv_proj = Conv(c1, c2, 1)
        self.q_proj = Conv(c2, c2, 1)
        self.norm_q = nn.LayerNorm(c2)
        self.norm_kv = nn.LayerNorm(c2)
        self.norm_out = nn.LayerNorm(c2)
        self.out_proj = Conv(c2, c3, 1)
        self._c2 = c2
        self.scale = self._c2**-0.5

    def _apply_ln(self, x, norm):
        """Apply LayerNorm on the channel dimension of a (B, C, H, W) tensor."""
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        """Forward pass fusing two multi-scale feature maps.

        Args:
            x (list[torch.Tensor]): List of [x_kv, x_q].
                x_kv: (B, c1, H_kv, W_kv) key/value features.
                x_q:  (B, c2, H_q, W_q) query features; output resolution matches this.

        Returns:
            (torch.Tensor): Fused features of shape (B, c3, H_q, W_q).
        """
        x_kv, x_q = x
        B = x_q.shape[0]
        H_q, W_q = x_q.shape[2], x_q.shape[3]
        H_kv, W_kv = x_kv.shape[2], x_kv.shape[3]
        C = self._c2
        kv = self.kv_proj(x_kv)
        q = self.q_proj(x_q)

        q = self._apply_ln(q, self.norm_q)
        kv = self._apply_ln(kv, self.norm_kv)
        if H_kv > H_q:
            factor_h = H_kv // H_q
            factor_w = W_kv // W_q
            kv = kv.reshape(B, C, H_q, factor_h, W_q, factor_w)
            kv = kv.permute(0, 2, 4, 3, 5, 1).reshape(
                B, H_q, W_q, factor_h * factor_w, C
            )
            q = q.permute(0, 2, 3, 1).unsqueeze(3)  # (B, H_q, W_q, 1, C)
        else:
            factor_h = H_q // H_kv
            factor_w = W_q // W_kv
            q = q.reshape(B, C, H_kv, factor_h, H_kv, factor_w)
            q = q.permute(0, 2, 4, 3, 5, 1).reshape(
                B, H_kv, H_kv, factor_h * factor_w, C
            )
            kv = kv.permute(0, 2, 3, 1).unsqueeze(3)  # (B, H_kv, W_kv, 1, C)

        attn = torch.matmul(q, kv.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = (
            torch.matmul(attn, kv)
            .reshape(B, H_q, W_q, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # Residual connection + LayerNorm + output projection
        out = self._apply_ln(out + x_q, self.norm_out)
        return self.out_proj(out)


class FastLocalDETR(nn.Module):
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

    def forward(self, x):
        B, C, H, W = x.shape
        ks, Np = self.kernel_size, self.kernel_size**2

        # 1. Norm 依然需要
        x = (
            self.input_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        )  # (B, C, H, W)

        center_x = x.flatten(2).transpose(1, 2)  # (B, L, C)

        for layer in self.layers:
            # 核心优化：单次 Unfold，直接处理 QKV
            qkv = layer["qkv_conv"](x)  # (B, 3C, H, W)
            # 一次映射 + 一次滑动窗口 = 最小显存开销
            qkv_nb = self.unfold(qkv).view(B, 3, C, Np, H * W)
            q_nb = qkv_nb[:, 0].permute(0, 3, 1, 2).reshape(-1, Np, C)  # (B*L, Np, C)
            k_nb = qkv_nb[:, 1].permute(0, 3, 1, 2).reshape(-1, Np, C)
            v_nb = qkv_nb[:, 2].permute(0, 3, 1, 2).reshape(-1, Np, C)

            # --- SA ---
            q_nb = q_nb.view(-1, Np, self.head_num, C // self.head_num).transpose(
                1, 2
            )  # (B*L, Np,head_num，C//head_num) -> (B*L, head_num, Np, C//head_num)
            k_nb = k_nb.view(-1, Np, self.head_num, C // self.head_num).transpose(1, 2)
            attn = (q_nb @ k_nb.transpose(-2, -1)) * ((C // self.head_num) ** -0.5)

            attn = F.softmax(attn, dim=-1)

            v_nb = v_nb.view(-1, Np, self.head_num, C // self.head_num).transpose(1, 2)
            nb_encoded = (attn @ v_nb).transpose(1, 2).reshape(-1, Np, C)
            nb_encoded = layer["norm_sa"](nb_encoded)  # 这里简化了残差以测试速度

            # --- CA ---
            qc = (
                layer["q_center"](center_x)
                .view(-1, 1, self.head_num, C // self.head_num)
                .transpose(1, 2)
            )
            kc = nb_encoded.view(-1, Np, self.head_num, C // self.head_num).transpose(
                1, 2
            )
            vc = kc  # 简化演示

            attn_c = F.softmax((qc @ kc.transpose(-2, -1)) * (C**-0.5), dim=-1)
            ca_out = (attn_c @ vc).transpose(1, 2).reshape(B, H * W, C)

            center_x = layer["norm_ca"](center_x + ca_out)
            center_x = layer["norm_ffn"](center_x + layer["ffn"](center_x))

            if self.N > 1:
                x = center_x.transpose(1, 2).view(B, C, H, W).contiguous()

        return self.proj(center_x).transpose(1, 2).view(B, self.cout, H, W)


class FastLocalDETRBlockBackup(nn.Module):
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
        self.input_norm = LayerNorm2d(c)
        self.proj = nn.Conv(c, cout, 1) if c != cout else nn.Identity()

        # 核心点：用 1x1 卷积代替 Linear，并放在最前面
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=sight)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "qkv_conv": Conv(c, 3 * c, 1),  # 一次性完成邻居的 QKV
                        "norm_sa": nn.LayerNorm(
                            c // head_num,
                        ),
                        "q_center": nn.Linear(c, c),
                        "norm_ca": LayerNorm2d(
                            c,
                        ),
                        "ffn": nn.Sequential(Conv(c, c * 4, 1), Conv(c * 4, c, 1)),
                        "norm_ffn": LayerNorm2d(c),
                    }
                )
                for _ in range(N)
            ]
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ks, Np = self.kernel_size, self.kernel_size**2

        center_x = self.input_norm(x)
        # .permute(
        #    0, 2, 3, 1
        # )  # (B, H, W, C)  # (B, C, H, W) -> (B,H*W,self.head_num,C//self.head_num)

        for layer in self.layers:
            qkv = layer["qkv_conv"](center_x)  # (B, 3C, H, W)
            qkv_nb = (
                self.unfold(qkv)
                .view(B, 3, self.head_num, C // self.head_num, Np, H * W)
                .permute(0, 1, 5, 2, 4, 3)
                .contiguous()
            )  # (B, 3, L,head_num, NP, C // self.head_num)
            q_nb = qkv_nb[:, 0]
            k_nb = qkv_nb[:, 1]
            v_nb = qkv_nb[:, 2]

            attn = (q_nb @ k_nb.transpose(-2, -1)) * ((C // self.head_num) ** -0.5)

            attn = F.softmax(attn, dim=-1)

            nb_encoded = attn @ v_nb
            nb_encoded = layer["norm_sa"](nb_encoded + q_nb)

            # --- CA ---
            qc = layer["q_center"](
                center_x.reshape(B, C, H * W).transpose(1, 2)
            ).reshape(B, H * W, self.head_num, 1, C // self.head_num)
            kc = nb_encoded
            vc = kc

            attn_c = F.softmax(
                (qc @ kc.transpose(-2, -1)) * ((C // self.head_num) ** -0.5), dim=-1
            )
            ca_out = attn_c @ vc
            ca_out = ca_out.permute(0, 2, 3, 4, 1).reshape(B, C, H, W)

            center_x = layer["norm_ca"](center_x + ca_out)
            center_x = layer["norm_ffn"](center_x + layer["ffn"](center_x))

        return self.proj(center_x)


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
        self.input_norm = LayerNorm2d(c)
        self.proj = nn.Conv(c, cout, 1) if c != cout else nn.Identity()

        # 核心点：用 1x1 卷积代替 Linear，并放在最前面
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=sight)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "qkv_conv": Conv(c, 3 * c, 1),  # 一次性完成邻居的 QKV
                        "norm_sa": nn.LayerNorm(
                            c // head_num,
                        ),
                        "q_center": nn.Linear(c, c),
                        "norm_ca": LayerNorm2d(
                            c,
                        ),
                        "ffn": nn.Sequential(Conv(c, c * 4, 1), Conv(c * 4, c, 1)),
                        "norm_ffn": LayerNorm2d(c),
                    }
                )
                for _ in range(N)
            ]
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ks, Np = self.kernel_size, self.kernel_size**2

        center_x = self.input_norm(x)
        # .permute(
        #    0, 2, 3, 1
        # )  # (B, H, W, C)  # (B, C, H, W) -> (B,H*W,self.head_num,C//self.head_num)

        for layer in self.layers:
            qkv = layer["qkv_conv"](center_x)  # (B, 3C, H, W)
            qkv_nb = (
                self.unfold(qkv)
                .permute(0, 2, 1)
                .reshape(-1, 3, self.head_num, Np, C // self.head_num)
                # .view(B, 3, self.head_num, C // self.head_num, Np, H * W)
                # .permute(0, 1, 5, 2, 4, 3)
                # .contiguous()
            )  # (B, 3, L,head_num, NP, C // self.head_num)
            q_nb = qkv_nb[:, 0]
            k_nb = qkv_nb[:, 1]
            v_nb = qkv_nb[:, 2]

            attn = (q_nb @ k_nb.transpose(-2, -1)) * ((C // self.head_num) ** -0.5)

            attn = F.softmax(attn, dim=-1)

            nb_encoded = attn @ v_nb
            nb_encoded = layer["norm_sa"](nb_encoded + q_nb)

            # --- CA ---
            qc = layer["q_center"](
                center_x.reshape(B, C, H * W).transpose(1, 2)
            ).reshape(-1, self.head_num, 1, C // self.head_num)
            kc = nb_encoded
            vc = kc

            attn_c = F.softmax(
                (qc @ kc.transpose(-2, -1)) * ((C // self.head_num) ** -0.5), dim=-1
            )
            ca_out = attn_c @ vc
            ca_out = ca_out.reshape(B, H * W, C).permute(0, 2, 1).reshape(B, C, H, W)

            center_x = layer["norm_ca"](center_x + ca_out)
            center_x = layer["norm_ffn"](center_x + layer["ffn"](center_x))

        return self.proj(center_x)


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
        ks, Np = self.kernel_size, self.kernel_size**2

        # 1. Norm 依然需要
        x = (
            self.input_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        )  # (B, C, H, W)

        pos_embed = self.row_pos_embed + self.col_pos_embed  # (Np, C)
        center_x = x.flatten(2).transpose(1, 2)  # (B, L, C)
        center_x = center_x + pos_embed[(Np - 1) // 2, :]  # 添加中心位置编码

        for i, layer in enumerate(self.layers):
            # 核心优化：单次 Unfold，直接处理 QKV
            qkv = layer["qkv_conv"](x)  # (B, 3C, H, W)
            # 一次映射 + 一次滑动窗口 = 最小显存开销
            qkv_nb = (
                self.unfold(qkv)
                .view(B, 3, C, Np, H * W)
                .permute(0, 1, 4, 3, 2)
                .contiguous()
            )  # (B, 3, L, NP, C)

            # 只在第一层循环时添加位置编码
            if i == 0:
                qkv_nb = qkv_nb + pos_embed

            # --- 优化：分批处理矩阵乘法以节省显存 (B*L 拆分为逐个 batch 处理) ---
            # 本来是 (B*L, head, Np, Np)，当 B*L 很大时 (例如 16*6400=102400) 很容易 OOM
            # 我们将 B*L 还原为 (B, L, ...) 然后逐个样本或逐个 head 处理是不够的
            # 这里我们简单的将 QKV 保持 (B, L, ...) 格式，每次只处理一个 Batch 的全量像素

            # 分离 QKV (B, L, Np, C)
            L = H * W
            q_nb_all = qkv_nb[:, 0]  # (B, L, Np, C)
            k_nb_all = qkv_nb[:, 1]
            v_nb_all = qkv_nb[:, 2]

            ca_out_list = []
            for b in range(B):
                # 针对当前 batch 的 L 个位置进行处理
                qb = (
                    q_nb_all[b]
                    .view(L, Np, self.head_num, C // self.head_num)
                    .transpose(1, 2)
                )  # (L, head, Np, d)
                kb = (
                    k_nb_all[b]
                    .view(L, Np, self.head_num, C // self.head_num)
                    .transpose(1, 2)
                )
                vb = (
                    v_nb_all[b]
                    .view(L, Np, self.head_num, C // self.head_num)
                    .transpose(1, 2)
                )

                # --- SA ---
                attn = (qb @ kb.transpose(-2, -1)) * ((C // self.head_num) ** -0.5)
                attn = F.softmax(attn, dim=-1)
                nb_encoded_b = (
                    (attn @ vb).transpose(1, 2).reshape(L, Np, C)
                )  # (L, Np, C)
                nb_encoded_b = layer["norm_sa"](nb_encoded_b)

                # --- CA ---
                # qcb: (L, head, 1, d)
                qcb = (
                    layer["q_center"](center_x[b])
                    .view(L, 1, self.head_num, C // self.head_num)
                    .transpose(1, 2)
                )
                kcb = nb_encoded_b.view(
                    L, Np, self.head_num, C // self.head_num
                ).transpose(1, 2)
                vcb = kcb  # 简化

                attn_cb = F.softmax((qcb @ kcb.transpose(-2, -1)) * (C**-0.5), dim=-1)
                ca_out_b = (attn_cb @ vcb).transpose(1, 2).reshape(L, C)  # (L, C)
                ca_out_list.append(ca_out_b)

            ca_out = torch.stack(ca_out_list, dim=0)  # (B, L, C)

            center_x = layer["norm_ca"](center_x + ca_out)
            center_x = layer["norm_ffn"](center_x + layer["ffn"](center_x))

            x = center_x.transpose(1, 2).view(B, C, H, W).contiguous()

        return self.proj(center_x).transpose(1, 2).view(B, self.cout, H, W)


if __name__ == "__main__":
    import time

    # 设定参数
    C = 64
    H = W = 80
    sight = 2
    head_num = 2
    N = 2
    iterations = 100  # 循环次数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模块
    block_a = FastLocalDETRBlock(C, C, sight=sight, head_num=head_num, N=N).to(device)
    block_b = FastLocalDETR(C, C, sight=sight, head_num=head_num, N=N).to(device)

    x = torch.randn((1, C, H, W)).to(device)

    # 预热
    print("Warming up...")
    for _ in range(10):
        _ = block_a(x)
        _ = block_b(x)

    torch.cuda.synchronize()

    # 测试 FastLocalDETRBlock (使用 pipeline_a 风格逻辑的模块)
    print(f"Testing FastLocalDETRBlock over {iterations} iterations...")
    t0 = time.time()
    for _ in range(iterations):
        _ = block_a(x)
    torch.cuda.synchronize()
    t1 = time.time()
    avg_a = (t1 - t0) / iterations * 1000
    print(f"FastLocalDETRBlock (Correct logic) average time: {avg_a:.3f} ms")

    # 测试 FastLocalDETR (使用 pipeline_b 风格逻辑的模块)
    print(f"Testing FastLocalDETR over {iterations} iterations...")
    t0 = time.time()
    for _ in range(iterations):
        _ = block_b(x)
    torch.cuda.synchronize()
    t1 = time.time()
    avg_b = (t1 - t0) / iterations * 1000
    print(f"FastLocalDETR (Incorrect logic) average time: {avg_b:.3f} ms")

    # 计算 FLOPs 和 Params
    from thop import profile

    flops_a, params_a = profile(block_a, inputs=(x,), verbose=False)
    flops_b, params_b = profile(block_b, inputs=(x,), verbose=False)
    print("-" * 30)
    print(f"Block A - FLOPs: {flops_a/1e9:.3f} G, Params: {params_a/1e6:.3f} M")
    print(f"Block B - FLOPs: {flops_b/1e9:.3f} G, Params: {params_b/1e6:.3f} M")
    print("-" * 30)
