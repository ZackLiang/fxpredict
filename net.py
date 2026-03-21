from layer import *
import math

# ══════════════════════════════════════════════════════════════════════
#  Mamba 时序模块（替换扩张卷积 TCN，增强长程依赖）
#
#  动机：扩张卷积感受野固定，而 Mamba 的选择性状态空间能根据信号强度
#        动态调整记忆权重，更适合 168 步长序列的宏观周期捕捉。
#
#  实现：轻量级 SelectiveSSM（纯 PyTorch），兼容 CPU/MPS/CUDA；
#        若安装 mamba_ssm 则优先使用官方实现。
# ══════════════════════════════════════════════════════════════════════

def _try_import_mamba():
    """尝试导入 mamba_ssm，失败则返回 None"""
    try:
        from mamba_ssm import Mamba
        return Mamba
    except ImportError:
        return None

_MambaOfficial = _try_import_mamba()


class SelectiveSSM(nn.Module):
    """
    轻量级选择性状态空间模块（Selective SSM），纯 PyTorch 实现。
    输入依赖的 gate 实现选择性记忆：h_t = (1-g)*h_{t-1} + g*x_t，
    可并行化为 cumprod/cumsum，适应不同强度的外汇信号。
    兼容 CPU/MPS/CUDA，无需 CUDA 内核。
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.gate_proj = nn.Linear(self.d_inner, 1)
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) 因果序列
        return: (B, T, D)
        选择性递推: m_t = (1-g_t)*m_{t-1} + g_t*h_t。
        采用标准递推避免 cumprod 在 T=168 时 float32 下溢（0.5^168→0）。
        """
        B, T, D_dim = x.shape
        h = self.in_proj(x)
        h, gate_h = h.chunk(2, dim=-1)
        h = h * torch.sigmoid(gate_h)
        gate = torch.sigmoid(self.gate_proj(h)).clamp(0.02, 0.98)  # (B, T, 1)

        m = torch.zeros(B, self.d_inner, device=x.device, dtype=x.dtype)
        out = []
        for t in range(T):
            g_t = gate[:, t, :]  # (B, 1)
            h_t = h[:, t, :]     # (B, d_inner)
            m = (1 - g_t) * m + g_t * h_t
            out.append(m)
        m_seq = torch.stack(out, dim=1)
        return self.out_proj(m_seq)


class MambaLayer(nn.Module):
    """
    Mamba 时序模块，替换 dilated_inception。
    接口与 dilated_inception 一致：cin, cout, dilation_factor。
    支持多尺度：沿时间维处理，输出长度与 dilated_inception 对齐
    （裁剪最后 6*dilation 步，匹配 kernel=7 的因果卷积）。
    """
    def __init__(self, cin: int, cout: int, dilation_factor: int = 1):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.dilation_factor = dilation_factor
        self.crop = 6 * max(1, dilation_factor)  # 与 dilated_inception kernel=7 对齐

        if _MambaOfficial is not None:
            d_model = cin
            self.use_official = True
            self.mamba = _MambaOfficial(
                d_model=d_model, d_state=16, d_conv=4, expand=2
            )
            self.out_proj = nn.Conv2d(d_model, cout, kernel_size=(1, 1))
        else:
            self.use_official = False
            self.ssm = SelectiveSSM(d_model=cin, d_state=16, expand=2)
            self.out_proj = nn.Conv2d(cin, cout, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, N, T)
        return: (B, C_out, N, T')  where T' = T - self.crop
        """
        B, C, N, T = x.shape
        if T <= self.crop:
            return torch.zeros(B, self.cout, N, 1, device=x.device, dtype=x.dtype)
        # reshape: (B, C, N, T) -> (B*N, T, C)
        x = x.permute(0, 2, 3, 1).reshape(B * N, T, C)
        if self.use_official:
            out = self.mamba(x)  # (B*N, T, C)
        else:
            out = self.ssm(x)    # (B*N, T, C)
        # 裁剪以匹配 dilated_inception 的输出长度
        out = out[:, self.crop:, :]  # (B*N, T', C)
        T_out = out.size(1)
        # reshape: (B*N, T', C) -> (B, C, N, T')，切片后需 contiguous 才能送入 Conv2d
        out = out.reshape(B, N, T_out, C).permute(0, 3, 1, 2).contiguous()
        return self.out_proj(out)


# ══════════════════════════════════════════════════════════════════════
#  双图时序交叉注意力融合 (Dual-Graph Temporal Cross-Attention)
#  ── 借鉴 Crossformer (ICLR 2023) 跨维度注意力 ───────────────────────
#
#  设计：Q 来自动态图，K/V 来自格兰杰图，在 T 维做注意力；融合 = gate*Attn + (1-gate)*x_A
# ══════════════════════════════════════════════════════════════════════
class DualGraphCrossAttn(nn.Module):
    """
    双图时序交叉注意力融合模块。
    将每个节点的时间序列特征视为独立序列，在时间维度(T)上做单头注意力。

    输入：
        x_A: [B, C, N, T]  动态图产生的节点特征
        x_B: [B, C, N, T]  静态先验图（格兰杰图）产生的节点特征
    输出：
        fused: [B, C, N, T]  融合后特征
    """
    def __init__(self, channels: int):
        super().__init__()
        d = max(channels // 2, 4)   # 压缩维度，降参数量
        # Q 来自动态图（时间步级别查询），K/V 来自格兰杰图
        self.q_proj = nn.Linear(channels, d, bias=False)
        self.k_proj = nn.Linear(channels, d, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        # 输出门控：每个节点×时间步独立决定注入多少来自格兰杰图的信息
        # 【Fix 8 v2】gate bias=0.0 → 初始 gate=sigmoid(0)=0.50（中性启动）
        # 原 bias=-2.0（gate≈0.12）与 Router bias=2.0（alpha≈0.88）双重叠加，
        # 实际格兰杰影响 ≈ (1-0.88) × 0.12 ≈ 1.4%，20 epoch 完全无法激活。
        # 中性启动：gate 初始为 0.5，由数据驱动学习何时多用格兰杰信息；
        # weight 保持零初始化，确保训练初期 gate 仅由 bias 决定（稳定启动）。
        _gate_linear = nn.Linear(channels * 2, 1, bias=True)
        nn.init.zeros_(_gate_linear.weight)
        nn.init.constant_(_gate_linear.bias, 0.0)
        self.gate   = nn.Sequential(_gate_linear, nn.Sigmoid())
        self.scale  = d ** -0.5

    def forward(self, x_A, x_B):
        B, C, N, T = x_A.shape
        # reshape: [B, C, N, T] → [B*N, T, C]  每个节点独立处理
        xA = x_A.permute(0, 2, 3, 1).reshape(B * N, T, C)   # [B*N, T, C]
        xB = x_B.permute(0, 2, 3, 1).reshape(B * N, T, C)   # [B*N, T, C]

        Q = self.q_proj(xA)   # [B*N, T, d]
        K = self.k_proj(xB)   # [B*N, T, d]
        V = self.v_proj(xB)   # [B*N, T, C]

        # 时序注意力：T×T，每个时间步可从格兰杰图任意时间步聚合信息
        attn = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)  # [B*N, T, T]
        cross = torch.bmm(attn, V)   # [B*N, T, C]

        # 门控残差融合：每个(节点,时间步)独立决定信息注入量
        g = self.gate(torch.cat([xA, cross], dim=-1))   # [B*N, T, 1]
        fused = xA + g * (cross - xA)                   # [B*N, T, C]

        # reshape 回原始格式 [B, C, N, T]
        return fused.reshape(B, N, T, C).permute(0, 3, 1, 2).contiguous()


# ══════════════════════════════════════════════════════════════════════
#  体制感知混合专家路由 (RegimeMoE)
#
#  三专家设计：
#    Trend Expert：动态自适应图，优化趋势行情预测
#    Range Expert：CrossAttn 融合，优化震荡回归行情预测
#    Granger Expert：预定义格兰杰图，专门处理结构化因果信号
#
#  傅里叶特征：FFT 幅值识别市场周期性，辅助体制判断
#  Gate：Softmax 3 维权重，对三专家加权融合
# ══════════════════════════════════════════════════════════════════════
NUM_EXPERTS = 3  # Trend, Range, Granger
FFT_TOP_K = 8   # 取 FFT 幅值前 k 个低频分量


class RegimeMoERouter(nn.Module):
    """
    体制感知混合专家路由器（RegimeMoE）。
    输入：RevIN 归一化序列 + FFT 幅值特征
    输出：Softmax 3 维权重 [w_trend, w_range, w_granger]
    """
    SHORT_WIN = 8
    MID_WIN   = 24

    def __init__(self, in_dim: int, hidden: int = 48):
        super().__init__()
        # 波动率特征：8*in_dim（同原 MultiScaleVolatilityRouter）
        # 傅里叶特征：FFT 幅值（截面平均），取 top-k 低频分量
        # 每个 in_dim 通道做 FFT，截面平均后取前 FFT_TOP_K 个幅值 → in_dim * FFT_TOP_K
        fft_dim = in_dim * FFT_TOP_K
        vol_dim = 8 * in_dim
        feat_dim = vol_dim + fft_dim
        self.bn = nn.BatchNorm1d(feat_dim)  # 跨 batch 归一化，保留 ratio/vol 尺度信息
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, NUM_EXPERTS),
        )
        with torch.no_grad():
            self.mlp[-1].bias.fill_(0.0)

    def _fft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim, N, seq_len]
        return: [B, in_dim * FFT_TOP_K]  截面平均后的 FFT 幅值（取低频）
        """
        B, D, N, T = x.shape
        # [B, D, N, T] -> 沿 T 做 FFT
        X = torch.fft.rfft(x, dim=-1)  # [B, D, N, T//2+1]
        mag = torch.abs(X)             # [B, D, N, n_freq]
        # 截面平均 [B, D, n_freq]
        mag = mag.mean(dim=2)
        # 取前 FFT_TOP_K 个低频分量（除 DC 外，低频=周期长=趋势）
        k = min(FFT_TOP_K, mag.size(-1))
        mag = mag[..., :k]  # [B, D, k]
        return mag.reshape(B, -1)

    def forward(self, x_normed: torch.Tensor) -> torch.Tensor:
        """
        x_normed: [B, in_dim, N, seq_len]
        return: [B, 3]  logits，外部做 Softmax
        """
        seq_len = x_normed.size(-1)
        short_w = min(self.SHORT_WIN, seq_len)
        mid_w   = min(self.MID_WIN, seq_len)

        # ── 波动率特征（与原 MultiScaleVolatilityRouter 一致）──
        vs_per_node = x_normed[..., -short_w:].std(dim=-1)
        vm_per_node = x_normed[..., -mid_w:  ].std(dim=-1)
        vl_per_node = x_normed.std(dim=-1)
        vol_short   = vs_per_node.mean(dim=-1)
        vol_mid     = vm_per_node.mean(dim=-1)
        vol_long    = vl_per_node.mean(dim=-1)
        disp_short  = vs_per_node.std(dim=-1)
        disp_mid    = vm_per_node.std(dim=-1)
        disp_long   = vl_per_node.std(dim=-1)
        ratio_sl    = vol_short / (vol_long + 1e-8)
        ratio_ml    = vol_mid   / (vol_long + 1e-8)
        feat_vol    = torch.cat([
            vol_short, vol_mid, vol_long,
            disp_short, disp_mid, disp_long,
            ratio_sl, ratio_ml,
        ], dim=-1)

        # ── 傅里叶特征（识别周期性趋势）──
        feat_fft = self._fft_magnitude(x_normed)

        # ── 拼接后做 BatchNorm（跨 batch 归一化，避免 dim=-1 对小维坍缩）
        feat = torch.cat([feat_vol, feat_fft], dim=-1)  # [B, feat_dim]
        feat = self.bn(feat)
        return self.mlp(feat)  # [B, 3]


class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device,
                 predefined_A=None, static_feat=None, dropout=0.3,
                 subgraph_size=20, node_dim=40, dilation_exponential=1,
                 conv_channels=32, residual_channels=32, skip_channels=64,
                 end_channels=128, seq_length=12, in_dim=2, out_dim=12,
                 layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True,
                 revin=True, dual_graph=True, use_router=True,
                 use_cross_attn=True, use_mamba=True):
        """
        新增参数：
            use_cross_attn (bool): use_router=True 时启用 DualGraphCrossAttn；
                                   设为 False 时退化为标量融合。
            use_mamba (bool): 1=MambaLayer（创新点1），0=原始 dilated_inception（TCN，消融 M0 用）
        """
        super(gtnet, self).__init__()
        self.gcn_true      = gcn_true
        self.buildA_true   = buildA_true
        self.num_nodes     = num_nodes
        self.dropout       = dropout
        self.device        = device
        self.use_cross_attn = use_cross_attn
        self.use_mamba     = use_mamba

        # ── predefined_A：注册为 buffer，随 model.to(device) 自动迁移 ──────
        # 直接赋值为普通属性时，调用 model.to(mps) 不会自动迁移 Tensor，
        # 导致 forward 中 predefined_A（cpu）与 x（mps）设备不一致而崩溃。
        # 用 register_buffer 保证设备始终与模型一致。
        #
        # 同时预计算行归一化转置矩阵（predefined_A_T）：
        #   predefined_A 是行归一化的格兰杰图（每行和=1）；
        #   其转置 predefined_A.T 的列和=1，但行和不等于1，
        #   在 granger_nconv 中做消息传递时会导致数值不稳定（部分节点聚合超大权重）。
        #   预计算行归一化版本，确保正向/反向两路聚合数值量级一致。
        if predefined_A is not None:
            self.register_buffer('predefined_A', predefined_A)
            # 转置后行归一化（保证反向格兰杰路径的数值稳定性）
            A_T = predefined_A.t().contiguous()
            row_sums_T = A_T.sum(dim=1, keepdim=True).clamp(min=1e-8)
            self.register_buffer('predefined_A_T', A_T / row_sums_T)
        else:
            self.predefined_A = None
            self.predefined_A_T = None

        self.filter_convs   = nn.ModuleList()
        self.gate_convs     = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs     = nn.ModuleList()
        self.gconv1         = nn.ModuleList()   # Expert A：动态自适应图 GCN
        self.gconv2         = nn.ModuleList()
        self.norm           = nn.ModuleList()

        # Expert B：静态先验图 GCN
        self.expert_gconv1  = nn.ModuleList()
        self.expert_gconv2  = nn.ModuleList()
        # 格兰杰图直接聚合投影（修复 mixprop 度归一化稀释问题）
        self.granger_proj   = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim,
                                    device, alpha=tanhalpha,
                                    static_feat=static_feat)
        self.seq_length = seq_length

        # RevIN 归一化 ────────────────────────────────────────────────
        self.revin_enabled = revin
        if self.revin_enabled:
            self.revin = RevIN(num_nodes, affine=True)

        # 双图 & 路由器 共用标志 ─────────────────────────────────────
        self.dual_graph = dual_graph
        self.use_router = use_router

        # 初始化路由器 & 融合模块（供外部脚本读取，需在 __init__ 中声明）
        self.last_alpha     = None
        self.last_fusion_w  = None
        self.last_adp       = None

        if self.dual_graph:
            if self.use_router:
                # RegimeMoE 体制感知混合专家路由 ────────────────────────
                # 3 专家 + FFT 特征 + Softmax 门控
                self.router = RegimeMoERouter(in_dim=in_dim, hidden=48)
            else:
                # 标量融合：可学习权重
                # 【Fix 1】初始 logit=3.0 → w=sigmoid(3)≈0.95
                # 动态图占95%，静态图仅贡献5%，避免随机初始化的静态图分支
                # 在训练早期注入大量噪声破坏动态图已收敛的表示。
                # 随训练深入，logit 会自动调整到最优值。
                self.fusion_logit = nn.Parameter(
                    torch.tensor(3.0, dtype=torch.float32), requires_grad=True)

        if self.dual_graph and self.predefined_A is None:
            print("Warning: Dual Graph is enabled but predefined_A is None!")

        # ── 感受野 & 各层模块 ─────────────────────────────────────────
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1)
                / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1)
                    * (dilation_exponential ** layers - 1)
                    / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1

            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size - 1)
                        * (dilation_exponential ** j - 1)
                        / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                TSeqLayer = MambaLayer if self.use_mamba else dilated_inception
                self.filter_convs.append(
                    TSeqLayer(residual_channels, conv_channels,
                             dilation_factor=new_dilation))
                self.gate_convs.append(
                    TSeqLayer(residual_channels, conv_channels,
                             dilation_factor=new_dilation))
                self.residual_convs.append(
                    nn.Conv2d(conv_channels, residual_channels,
                              kernel_size=(1, 1)))

                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(
                        nn.Conv2d(conv_channels, skip_channels,
                                  kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(
                        nn.Conv2d(conv_channels, skip_channels,
                                  kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                if self.gcn_true:
                    self.gconv1.append(
                        mixprop(conv_channels, residual_channels,
                                gcn_depth, dropout, propalpha))
                    self.gconv2.append(
                        mixprop(conv_channels, residual_channels,
                                gcn_depth, dropout, propalpha))
                    if self.dual_graph:
                        self.expert_gconv1.append(
                            mixprop(conv_channels, residual_channels,
                                    gcn_depth, dropout, propalpha))
                        self.expert_gconv2.append(
                            mixprop(conv_channels, residual_channels,
                                    gcn_depth, dropout, propalpha))
                        # ── 格兰杰图直接聚合投影 ────────────────────────────
                        # 问题：mixprop 在传播前做 adj += I 再度归一化，
                        #   格兰杰图行和已归一化为1，加I后行和≈2，
                        #   原始边权重被压缩约50%，结构信号严重衰减。
                        # 修复：额外一步直接聚合（无自环、无度归一化），
                        #   直接用归一化格兰杰图矩阵做一步消息传递，
                        #   输出通过 granger_proj 映射到 residual_channels，
                        #   以残差方式叠加到 x_B 上，保留格兰杰的完整结构先验。
                        # 【Fix 7】granger_proj 零初始化：
                        # 训练初期 granger_proj(x_granger_direct)≈0，
                        # 避免随机权重给 x_B 叠加额外噪声，
                        # 随梯度自然学习何时补充格兰杰直接聚合信号。
                        _gp = nn.Conv2d(conv_channels, residual_channels,
                                        kernel_size=(1, 1), bias=False)
                        nn.init.zeros_(_gp.weight)
                        self.granger_proj.append(_gp)

                if self.seq_length > self.receptive_field:
                    self.norm.append(
                        LayerNorm((residual_channels, num_nodes,
                                   self.seq_length - rf_size_j + 1),
                                  elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(
                        LayerNorm((residual_channels, num_nodes,
                                   self.receptive_field - rf_size_j + 1),
                                  elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels,
                                    kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim,
                                    kernel_size=(1, 1), bias=True)

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_dim, skip_channels,
                                   kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(residual_channels, skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1),
                                   bias=True)
        else:
            self.skip0 = nn.Conv2d(in_dim, skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(residual_channels, skip_channels,
                                   kernel_size=(1, 1), bias=True)

        # 双图时序交叉注意力融合 ─────────────────────────────────────
        # use_router=True 且 use_cross_attn=True 时启用
        # use_cross_attn=False 时使用标量融合
        if self.dual_graph and self.use_router and self.use_cross_attn:
            self.cross_attn = DualGraphCrossAttn(channels=residual_channels)

        # 格兰杰图直接聚合所需的 nconv 算子（无参数，用于直接消息传递）
        # 在 __init__ 中注册，避免 forward 循环中重复创建
        if self.dual_graph:
            self.granger_nconv = nconv()

        # register_buffer 保证 idx 随 model.to(device) 自动迁移（MPS/CUDA 均安全）
        self.register_buffer('idx', torch.arange(self.num_nodes))

    # ──────────────────────────────────────────────────────────────────
    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len == self.seq_length, \
            'input sequence length not equal to preset sequence length'

        # ── RevIN 归一化 ─────────────────────────────────────────────
        if self.revin_enabled:
            input, _ = self.revin(input, 'norm')

        # ── RegimeMoE Router（体制感知 3 专家门控）──────────────────────
        # 输出 [w_trend, w_range, w_granger] Softmax 权重
        if self.dual_graph and self.use_router:
            gate_logits = self.router(input)           # [B, 3]
            gate = F.softmax(gate_logits, dim=-1)      # [B, 3]
            self.last_alpha = gate.detach().cpu().numpy()
            gate = gate.view(-1, 3, 1, 1)             # [B, 3, 1, 1] 用于专家加权
        else:
            gate = None

        # ── Padding ──────────────────────────────────────────────────
        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(
                input, (self.receptive_field - self.seq_length, 0, 0, 0))

        # ── 动态自适应图构建 ──────────────────────────────────────────
        adp_learned = None
        if self.gcn_true:
            if self.buildA_true:
                adp_learned = self.gc(self.idx if idx is None else idx)
                # .cpu() 确保 MPS/CUDA 下也能 .numpy()（numpy 只支持 cpu tensor）
                self.last_adp = adp_learned.detach().float().cpu().numpy()
            else:
                adp_learned = self.predefined_A

        # ── 主干网络 ──────────────────────────────────────────────────
        x    = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x
            filter_  = torch.tanh(self.filter_convs[i](x))
            gate_    = torch.sigmoid(self.gate_convs[i](x))
            x = filter_ * gate_
            x = F.dropout(x, self.dropout, training=self.training)
            skip = self.skip_convs[i](x) + skip

            if self.gcn_true:
                # Expert A：动态自适应图（所有模型共用）
                x_A = (self.gconv1[i](x, adp_learned)
                       + self.gconv2[i](x, adp_learned.transpose(1, 0)))

                if self.dual_graph and self.predefined_A is not None:
                    # Expert B：静态先验图（独立 GCN 权重）
                    # 第一步：mixprop 提取多跳特征（mixprop内部自动重归一化，转置安全）
                    x_B_gcn = (self.expert_gconv1[i](x, self.predefined_A)
                               + self.expert_gconv2[i](x, self.predefined_A_T))

                    # 【修复】第二步：直接图聚合残差（保留格兰杰图原始结构信号）
                    # 问题：mixprop 内部做 adj += I 再度归一化，格兰杰图的有向边权
                    #   被压缩约50%，导致 x_B_gcn 中格兰杰结构信息严重衰减。
                    # 修复：granger_nconv 直接用行归一化格兰杰矩阵做一步消息传递（无自环），
                    #   通过 granger_proj 映射到 residual_channels 后，
                    #   以残差方式叠加到 x_B_gcn，恢复格兰杰图的完整结构先验。
                    # predefined_A: [N, N] 行归一化格兰杰矩阵（行和=1）
                    # predefined_A_T: 转置后的行归一化矩阵（保证数值稳定性）
                    # 双向聚合：正向（因果方向）+ 反向（被影响方向）
                    x_granger_direct = (
                        self.granger_nconv(x, self.predefined_A) +
                        self.granger_nconv(x, self.predefined_A_T)
                    )
                    x_B = x_B_gcn + self.granger_proj[i](x_granger_direct)

                    if self.use_router and gate is not None:
                        # RegimeMoE 三专家加权融合 ────────────────────────
                        # Trend Expert: x_A（动态图，趋势行情）
                        # Range Expert: x_fused（CrossAttn 融合，震荡回归）
                        # Granger Expert: x_B（格兰杰图，结构化因果信号）
                        if self.use_cross_attn:
                            x_fused = self.cross_attn(x_A, x_B)
                            x = gate[:, 0:1] * x_A + gate[:, 1:2] * x_fused + gate[:, 2:3] * x_B
                        else:
                            x_fused = x_B
                            x = gate[:, 0:1] * x_A + (gate[:, 1:2] + gate[:, 2:3]) * x_B
                    else:
                        # 标量融合：可学习残差注入
                        # 【Fix 2】改凸组合为残差注入：x = x_A + w * x_B
                        # 原凸组合 w*x_A + (1-w)*x_B：初始w=0.5时50%是随机噪声，
                        #   破坏x_A（动态图）的已收敛表示，导致前期MAE恶化。
                        # 残差注入：x_A永远完整保留，静态图仅贡献增量信号，
                        #   w初始≈0.95 → x_B贡献≈0.95倍，但x_A不被稀释；
                        #   等价于：模型先学会"动态图预测"，再学会"如何用格兰杰图修正"。
                        # w→0 退化为纯动态图，w>0 时格兰杰图有增量贡献。
                        w = torch.sigmoid(self.fusion_logit)
                        self.last_fusion_w = w.item()
                        x = x_A + w * x_B
                else:
                    x = x_A
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, self.idx if idx is None else idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # ── RevIN 反归一化 ────────────────────────────────────────────
        if self.revin_enabled:
            x = self.revin(x, 'denorm', target_idx=0)
        return x


# ══════════════════════════════════════════════════════════════════════
#  Baseline 模型库（对比实验用）
#
#  统一接口：输入 (B, 1, M, P)，输出 (B, M, 1, 1)。forward(x, idx=None)。
#  各 baseline 在 forward 内手动做 RevIN（_mean/_std 局部变量）。
# ══════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────
# 1. LSTMBaseline
#    经典双层 LSTM，输入窗口 → 隐状态 → 全连接输出
#    对照意义：证明 GNN 的图结构建模优于纯序列建模
# ──────────────────────────────────────────────────────────────────────
class LSTMBaseline(nn.Module):
    """
    双层 LSTM Baseline。
    输入:  X (B, 1, M, P) → reshape 为 (B, P, M)
    输出:  (B, M, 1, 1)
    """
    def __init__(self, num_nodes: int, seq_in_len: int,
                 hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, revin: bool = True):
        super().__init__()
        self.num_nodes   = num_nodes
        self.revin_on    = revin
        # RevIN 直接在 forward 里用局部变量计算（实例归一化），无需独立对象
        self.lstm = nn.LSTM(
            input_size=num_nodes, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, num_nodes)

    def forward(self, x, idx=None):
        # x: (B, 1, M, P)
        B, _, M, P = x.shape
        x = x.squeeze(1)                     # (B, M, P)
        if self.revin_on:
            # 实例归一化：对每个样本的时间维度统计，与主模型 RevIN 语义一致
            _mean = x.mean(dim=-1, keepdim=True).detach()  # (B, M, 1)
            _std  = x.std(dim=-1, keepdim=True).detach() + 1e-5  # (B, M, 1)
            x = (x - _mean) / _std
        x = x.permute(0, 2, 1)              # (B, P, M)
        out, _ = self.lstm(x)               # (B, P, H)
        pred = self.fc(out[:, -1, :])       # (B, M)
        if self.revin_on:
            # 反归一化：恢复原始尺度（squeeze 对齐维度）
            pred = pred * _std.squeeze(-1) + _mean.squeeze(-1)  # (B, M)
        return pred.unsqueeze(-1).unsqueeze(-1)   # (B, M, 1, 1)


# ──────────────────────────────────────────────────────────────────────
# 2. TCNBaseline
#    因果膨胀卷积（Temporal Convolutional Network），WaveNet 风格
#    输入窗口固定 P，感受野 = 2^(num_layers) × kernel_size
#    对照意义：证明图结构建模比纯卷积时序模型更有效
# ──────────────────────────────────────────────────────────────────────
class _CausalConv1d(nn.Module):
    """单层因果膨胀卷积（padding 保证输出长度 = 输入长度）"""
    def __init__(self, in_c, out_c, kernel_size, dilation):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_c, out_c, kernel_size,
                              dilation=dilation, padding=0)
        self.norm = nn.LayerNorm(out_c)
        self.act  = nn.GELU()
        # 残差投影（in_c ≠ out_c 时）
        self.res  = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        # x: (B*M, C, T)
        residual = self.res(x)
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x) + residual
        return x


class TCNBaseline(nn.Module):
    """
    4 层因果膨胀卷积 TCN。
    每个货币对独立处理（在 B*M 维展开），最后全连接输出。
    输入:  X (B, 1, M, P)
    输出:  (B, M, 1, 1)
    """
    def __init__(self, num_nodes: int, seq_in_len: int,
                 channels: int = 32, kernel_size: int = 3,
                 num_layers: int = 4, dropout: float = 0.1,
                 revin: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.revin_on  = revin
        # RevIN 直接在 forward 里用局部变量计算（实例归一化），无需独立对象
        layers = []
        in_c = 1
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(_CausalConv1d(in_c, channels, kernel_size, dilation))
            in_c = channels
        self.tcn = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc  = nn.Linear(channels, 1)   # 每个节点独立预测

    def forward(self, x, idx=None):
        # x: (B, 1, M, P)
        B, _, M, P = x.shape
        x = x.squeeze(1)                    # (B, M, P)
        if self.revin_on:
            # 实例归一化：沿时间维统计
            _mean = x.mean(dim=-1, keepdim=True).detach()  # (B, M, 1)
            _std  = x.std(dim=-1, keepdim=True).detach() + 1e-5
            x = (x - _mean) / _std
        # reshape 为 (B*M, 1, P)，每个节点独立卷积
        x = x.reshape(B * M, 1, P)          # (B*M, 1, P)
        x = self.tcn(x)                      # (B*M, C, P)
        x = self.dropout(x[:, :, -1])        # (B*M, C) 取最后一步
        pred = self.fc(x).squeeze(-1)        # (B*M,)
        pred = pred.reshape(B, M)            # (B, M)
        if self.revin_on:
            pred = pred * _std.squeeze(-1) + _mean.squeeze(-1)
        return pred.unsqueeze(-1).unsqueeze(-1)


# ──────────────────────────────────────────────────────────────────────
# 3. AGCRNBaseline
#    Adaptive Graph Convolutional Recurrent Network（NeurIPS 2020）
#    节点嵌入 E → 自适应邻接矩阵 A=softmax(E·E^T) → AGCN-GRU
#    对照意义：同为自适应图+RNN，无格兰杰先验，测试"静态先验的增量价值"
# ──────────────────────────────────────────────────────────────────────
class _AGCNCell(nn.Module):
    """单步 GRU + 自适应图卷积"""
    def __init__(self, in_dim, hidden_dim, num_nodes, emb_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes  = num_nodes
        # GRU gate projections（图卷积替代线性层）
        # r, u, c gate 各需要 (in+hidden) → hidden
        self.theta_r = nn.Linear(in_dim + hidden_dim, hidden_dim, bias=False)
        self.theta_u = nn.Linear(in_dim + hidden_dim, hidden_dim, bias=False)
        self.theta_c = nn.Linear(in_dim + hidden_dim, hidden_dim, bias=False)
        self.bn_r = nn.LayerNorm(hidden_dim)
        self.bn_u = nn.LayerNorm(hidden_dim)
        self.bn_c = nn.LayerNorm(hidden_dim)

    def forward(self, x, h, A):
        # x: (B, N, D_in)  h: (B, N, H)  A: (N, N) 自适应邻接矩阵
        # 图聚合：对 h 做图卷积 Ã·h
        Ah = torch.bmm(A.unsqueeze(0).expand(x.size(0), -1, -1), h)  # (B, N, H)
        Ax = torch.bmm(A.unsqueeze(0).expand(x.size(0), -1, -1), x)  # (B, N, D)
        inp_r = torch.cat([Ax, Ah], dim=-1)
        inp_u = inp_r
        r = torch.sigmoid(self.bn_r(self.theta_r(inp_r)))
        u = torch.sigmoid(self.bn_u(self.theta_u(inp_u)))
        inp_c = torch.cat([Ax, r * Ah], dim=-1)
        c = torch.tanh(self.bn_c(self.theta_c(inp_c)))
        h_new = u * h + (1.0 - u) * c
        return h_new


class AGCRNBaseline(nn.Module):
    """
    AGCRN（NeurIPS 2020）简化版。
    节点嵌入 E → 自适应图 A = softmax(ReLU(E·E^T))
    2 层 AGCN-GRU 沿时间展开，最后全连接输出。
    输入:  X (B, 1, M, P)
    输出:  (B, M, 1, 1)
    """
    def __init__(self, num_nodes: int, seq_in_len: int,
                 hidden_dim: int = 64, emb_dim: int = 10,
                 num_layers: int = 2, dropout: float = 0.1,
                 revin: bool = True):
        super().__init__()
        self.num_nodes  = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.revin_on   = revin
        # RevIN 直接在 forward 里用局部变量计算（实例归一化），无需独立对象
        # 节点嵌入（可学习）
        self.node_emb = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        # 多层 AGCN-GRU Cell
        self.cells = nn.ModuleList()
        in_d = 1
        for _ in range(num_layers):
            self.cells.append(_AGCNCell(in_d, hidden_dim, num_nodes, emb_dim))
            in_d = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def _build_A(self):
        """从节点嵌入构造自适应对称归一化邻接矩阵"""
        # A = softmax(ReLU(E·E^T))，行归一化
        A = torch.relu(torch.mm(self.node_emb, self.node_emb.t()))
        row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return A / row_sum   # (N, N)

    def forward(self, x, idx=None):
        # x: (B, 1, M, P)
        B, _, M, P = x.shape
        x = x.squeeze(1)                        # (B, M, P)
        if self.revin_on:
            # 实例归一化：沿时间维统计
            _mean = x.mean(dim=-1, keepdim=True).detach()  # (B, M, 1)
            _std  = x.std(dim=-1, keepdim=True).detach() + 1e-5
            x = (x - _mean) / _std
        x = x.permute(0, 2, 1)                  # (B, P, M)
        A = self._build_A()                     # (M, M)
        # 逐步展开
        hs = [torch.zeros(B, M, self.hidden_dim, device=x.device)
              for _ in range(self.num_layers)]
        for t in range(P):
            inp = x[:, t, :].unsqueeze(-1)      # (B, M, 1)
            for l, cell in enumerate(self.cells):
                hs[l] = cell(inp, hs[l], A)
                inp = self.dropout(hs[l])
        pred = self.fc(hs[-1]).squeeze(-1)      # (B, M)
        if self.revin_on:
            pred = pred * _std.squeeze(-1) + _mean.squeeze(-1)
        return pred.unsqueeze(-1).unsqueeze(-1)


# ──────────────────────────────────────────────────────────────────────
# 4. PatchTSTBaseline
#    PatchTST（ICLR 2023）简化版：把时序切 patch → Transformer Encoder
#    Channel-independent：每个货币对独立处理（原文默认设置）
#    对照意义：强 Transformer baseline，无图结构，测试"图先验 vs 纯注意力"
# ──────────────────────────────────────────────────────────────────────
class PatchTSTBaseline(nn.Module):
    """
    PatchTST（ICLR 2023）Channel-Independent 版本。
    将输入窗口切成 patch_len 大小的 patch，送入标准 Transformer Encoder。

    输入:  X (B, 1, M, P)
    输出:  (B, M, 1, 1)
    """
    def __init__(self, num_nodes: int, seq_in_len: int,
                 patch_len: int = 16, stride: int = 8,
                 d_model: int = 64, n_heads: int = 4,
                 num_encoder_layers: int = 3,
                 dropout: float = 0.1, revin: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.revin_on  = revin
        # RevIN 直接在 forward 里用局部变量计算（实例归一化），无需独立对象
        self.patch_len = patch_len
        self.stride    = stride
        # 序列切 patch 后的 patch 数量
        num_patches = (max(seq_in_len, patch_len) - patch_len) // stride + 1
        # Patch Embedding
        self.patch_emb = nn.Linear(patch_len, d_model)
        # 可学习位置编码
        self.pos_emb   = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        # Transformer Encoder（标准实现）
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True               # Pre-LN，训练更稳定
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.dropout  = nn.Dropout(dropout)
        self.head     = nn.Linear(d_model * num_patches, 1)

    def forward(self, x, idx=None):
        # x: (B, 1, M, P)
        B, _, M, P = x.shape
        x = x.squeeze(1)                    # (B, M, P)
        if self.revin_on:
            # 实例归一化：沿时间维统计（与主模型 RevIN 语义一致）
            _mean = x.mean(dim=-1, keepdim=True).detach()  # (B, M, 1)
            _std  = x.std(dim=-1, keepdim=True).detach() + 1e-5
            x = (x - _mean) / _std
        # Channel-Independent：reshape 为 (B*M, 1, P)，展开为 patch
        x = x.reshape(B * M, 1, P)
        # 提取 patch：unfold → (B*M, num_patches, patch_len)
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # (B*M, 1, n_patches, patch_len)
        x = x.squeeze(1)                    # (B*M, n_patches, patch_len)
        x = self.patch_emb(x)              # (B*M, n_patches, d_model)
        x = x + self.pos_emb[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.encoder(x)                # (B*M, n_patches, d_model)
        x = x.flatten(1)                   # (B*M, n_patches * d_model)
        pred = self.head(x).squeeze(-1)    # (B*M,)
        pred = pred.reshape(B, M)          # (B, M)
        if self.revin_on:
            pred = pred * _std.squeeze(-1) + _mean.squeeze(-1)
        return pred.unsqueeze(-1).unsqueeze(-1)


# ──────────────────────────────────────────────────────────────────────
# 5. iTransformerBaseline
#    iTransformer（ICLR 2024）简化版：倒置 Attention
#    把 Variate 维当 Token，在变量间做 self-attention 学相关性
#    Feed-Forward 在时间维做特征提取（与原文一致）
#    对照意义：2024 年 SOTA 无图 baseline，测试"显式图 vs 隐式变量注意力"
# ──────────────────────────────────────────────────────────────────────
class iTransformerBaseline(nn.Module):
    """
    iTransformer（ICLR 2024）。
    把 M 个变量作为 M 个 token，时间序列 P 作为 token 的特征维度。
    Multi-Head Self-Attention 在变量维捕捉跨变量依赖。
    Feed-Forward 在 token 内（时间维）做特征变换。

    输入:  X (B, 1, M, P)
    输出:  (B, M, 1, 1)
    """
    def __init__(self, num_nodes: int, seq_in_len: int,
                 d_model: int = 64, n_heads: int = 4,
                 num_layers: int = 3, d_ff: int = 256,
                 dropout: float = 0.1, revin: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.revin_on  = revin
        # RevIN 直接在 forward 里用局部变量计算（实例归一化），无需独立对象
        # 把每个 variate 的时间序列映射到 d_model 维 token embedding
        self.token_emb = nn.Linear(seq_in_len, d_model)
        # Transformer Encoder（在 variate 维 M 个 token 上做 attention）
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
            norm_first=True
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout  = nn.Dropout(dropout)
        # 输出投影：d_model → 1（单步预测）
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x, idx=None):
        # x: (B, 1, M, P)
        B, _, M, P = x.shape
        x = x.squeeze(1)                    # (B, M, P)
        if self.revin_on:
            # 实例归一化：每个样本沿时间维统计（与主模型 RevIN 语义一致）
            _mean = x.mean(dim=-1, keepdim=True).detach()   # (B, M, 1)
            _std  = x.std(dim=-1, keepdim=True).detach() + 1e-5
            x = (x - _mean) / _std
        tokens = self.token_emb(x)          # (B, M, d_model)  variate → token
        tokens = self.dropout(tokens)
        tokens = self.encoder(tokens)       # (B, M, d_model)  跨 variate attention
        pred   = self.proj(tokens).squeeze(-1)  # (B, M)
        if self.revin_on:
            pred = pred * _std.squeeze(-1) + _mean.squeeze(-1)
        return pred.unsqueeze(-1).unsqueeze(-1)
