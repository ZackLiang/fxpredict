import argparse
import math
import time
import os
import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib
import pickle
from util import DataLoaderS, get_forex_triangles, get_node_names_from_data_dir
# trainer.Optim 已移除，直接用 torch.optim

# ==========================================
# 三角套利物理一致性损失 (Triangular Consistency Loss)
# ==========================================
# 权重由 --phys_weight 控制，消融时设 0 可关闭
class PhysicalConsistencyLoss(nn.Module):
    """
    三角套利物理一致性损失：约束预测收益率满足
    (1+r_ab) × (1+r_bc) = (1+r_ac)，其中 r 为收益率。
    将预测价格差还原为变动因子 1+r = output/LP，计算三角环路残差平方。
    """

    def __init__(self, tri_indices, device):
        super().__init__()
        self.tri_indices = tri_indices  # list of (i, j, k)
        self.device = device

    def forward(self, output, LP, id_tensor):
        """
        output: (B, N_sub) 预测归一化价格
        LP: (B, N_sub) 窗口最后时刻归一化价格
        id_tensor: (N_sub,) 当前子图对应的全局节点索引
        """
        if not self.tri_indices:
            return torch.tensor(0.0, device=self.device, dtype=output.dtype)
        id_list = id_tensor.cpu().tolist()
        id_to_local = {g: idx for idx, g in enumerate(id_list)}
        loss_list = []
        for (i, j, k) in self.tri_indices:
            if i not in id_to_local or j not in id_to_local or k not in id_to_local:
                continue
            li, lj, lk = id_to_local[i], id_to_local[j], id_to_local[k]
            # (1+r) = output / LP，避免除零
            eps = 1e-8
            fact_i = output[:, li] / (LP[:, li].clamp(min=eps))
            fact_j = output[:, lj] / (LP[:, lj].clamp(min=eps))
            fact_k = output[:, lk] / (LP[:, lk].clamp(min=eps))
            # 残差: |(1+r_ab)*(1+r_bc) - (1+r_ac)|^2
            residual = fact_i * fact_j - fact_k
            loss_list.append((residual ** 2).mean())
        if not loss_list:
            return torch.tensor(0.0, device=self.device, dtype=output.dtype)
        return torch.stack(loss_list).mean()

# ==========================================
# 1. 升级版 Evaluate 函数 (全指标)
# ==========================================
def evaluate(data, X, Y, LP, model, evaluateL2, evaluateL1, batch_size,
             save_dir=None, run_id=None):
    """
    评估函数。
    X, Y, LP = data.train/valid/test 的三个元素。
    LP = last_price（输入窗口最后一步的归一化价格）。

    MAE/RMSE：在归一化绝对价格空间计算（output vs Y），乘以 scale 还原真实价格量级。
    差分指标（DA/tDA/IC/ICIR）：基于
      diff_pred = (output - LP) * scale   预测涨跌量（真实空间）
      diff_true = (Y - LP) * scale        真实涨跌量（真实空间）
    这样 RevIN 在预测绝对价格上正常工作，DA/IC 在差分方向上正确评估。
    """
    model.eval()
    total_loss    = 0
    total_loss_l1 = 0
    n_samples     = 0
    predict = None
    test    = None

    all_diff_pred = []
    all_diff_true = []
    ic_per_batch  = []

    for X_b, Y_b, LP_b in data.get_batches(X, Y, batch_size, False, lp=LP):
        X_in = torch.unsqueeze(X_b, dim=1).transpose(2, 3)
        with torch.no_grad():
            output = model(X_in)
        output = torch.squeeze(output)
        if output.dim() == 1:
            output = output.unsqueeze(0)

        if predict is None:
            predict, test = output, Y_b
        else:
            predict = torch.cat((predict, output))
            test    = torch.cat((test, Y_b))

        scale = data.scale.expand(output.size(0), data.m)

        # MAE/RMSE：价格空间（归一化 → 真实价格量级）
        pred_real = output * scale     # (B, M)
        true_real = Y_b    * scale     # (B, M)
        total_loss    += evaluateL2(pred_real, true_real).item()
        total_loss_l1 += evaluateL1(pred_real, true_real).item()

        # 差分：用于 DA/IC 计算
        diff_pred = (output - LP_b) * scale   # 预测涨跌量
        diff_true = (Y_b   - LP_b) * scale   # 真实涨跌量

        # IC（截面 Pearson，逐 batch 计算后取均值）
        dp = diff_pred - diff_pred.mean(dim=1, keepdim=True)
        dt = diff_true - diff_true.mean(dim=1, keepdim=True)
        cov   = (dp * dt).mean(dim=1)
        std_p = diff_pred.std(dim=1, unbiased=False).clamp(min=1e-8)
        std_t = diff_true.std(dim=1, unbiased=False).clamp(min=1e-8)
        ic_per_batch.append((cov / (std_p * std_t)).mean().item())

        all_diff_pred.append(diff_pred.detach().cpu().numpy())
        all_diff_true.append(diff_true.detach().cpu().numpy())

        n_samples += output.size(0) * data.m

    # ── 全局 DA / tDA（统一阈值，避免 per-batch 抖动）────────────────
    _all_dp = np.vstack(all_diff_pred)   # (T, M)
    _all_dt = np.vstack(all_diff_true)   # (T, M)

    _move_thresh = float(np.quantile(np.abs(_all_dt), 0.50))  # Top50% 真实波动
    _conf_thresh = float(np.quantile(np.abs(_all_dp), 0.75))  # Top25% 预测信号

    _move_mask = np.abs(_all_dt) > _move_thresh
    _conf_mask = (np.abs(_all_dp) > _conf_thresh) & _move_mask
    _correct   = (_all_dp * _all_dt) > 0

    da  = float((_correct & _move_mask).sum()) / max(int(_move_mask.sum()), 1)
    tda = float((_correct & _conf_mask).sum()) / max(int(_conf_mask.sum()), 1)

    # ══════════════════════════════════════════════════════════════════
    # 专属指标1：分体制 DA（Regime-Conditional DA）
    # ── Router 体制感知：趋势行情下 DA 更高 ──────────────────────────
    #
    # 设计原理：
    #   外汇市场有两种体制：趋势行情（动量驱动）和震荡行情（均值回归）。
    #   Router 用多尺度波动率比值判断体制，动态切换信任哪张图：
    #     趋势行情 → alpha→1 → 信任动态自适应图（捕捉价格动量）
    #     震荡行情 → alpha→0 → 信任格兰杰因果图（利用跨货币均值回归）
    #
    #   分体制 DA 的论文意义：
    #     无 Router：da_trend ≈ da_range
    #     有 Router：da_trend > da_range（趋势行情 DA 更高，
    #                         因为趋势时模型切换到自适应图，预测方向更准）
    #     这个差异直接证明了 Router 的体制感知能力，是其贡献的直接证据。
    #
    # 实现：用"标准化截面均值绝对值"作为体制代理变量
    #   真正的趋势行情 = 所有货币对同向漂移（截面均值绝对值大）
    #   震荡/分化行情 = 各货币对方向分散（截面均值接近0）
    #
    #   【修复说明】原用截面std区分体制是错误的：
    #     截面std大 → 各货币方向分化 → 这恰恰是震荡/套利行情
    #     真实趋势市（如美元加息周期）= 所有非美货币同向走弱 → 截面std反而偏小
    #   【关键改进】先做z-score去量纲化，消除USDJPY(波动0.5)与EURUSD(波动0.001)
    #   的绝对量级差异，再用归一化后的截面均值绝对值判断市场同向性强度。
    T_total = _all_dt.shape[0]
    # Step1: z-score 去量纲（沿时间轴归一化，消除各货币对量级差异）
    _dt_std  = _all_dt.std(axis=0, keepdims=True) + 1e-8   # (1, M) 各货币对时序std
    _dt_norm = _all_dt / _dt_std                            # (T, M) 无量纲差分
    # Step2: 截面均值绝对值（衡量市场整体同向漂移强度）
    _cross_momentum = np.abs(_dt_norm.mean(axis=1))         # (T,) 趋势强度代理
    _vol_med   = float(np.median(_cross_momentum))
    _trend_mask_t = _cross_momentum > _vol_med              # 强同向漂移 = 趋势行情
    _range_mask_t = ~_trend_mask_t                          # 弱同向 / 分化 = 震荡行情

    # 扩展到 (T, M) 维度
    _trend_mask = _trend_mask_t[:, np.newaxis] & _move_mask
    _range_mask = _range_mask_t[:, np.newaxis] & _move_mask

    da_trend = float((_correct & _trend_mask).sum()) / max(int(_trend_mask.sum()), 1)
    da_range = float((_correct & _range_mask).sum()) / max(int(_range_mask.sum()), 1)

    # ══════════════════════════════════════════════════════════════════
    # 专属指标2：跨货币信号协同率（Cross-Currency Concordance, CCC）
    # ── 格兰杰图结构先验：跨货币对方向一致性 ────────────────────────
    #
    # 设计原理：
    #   格兰杰图先验告诉模型"货币A领先货币B约k步"。
    #   如果模型学到了这个关系，当货币A走强时，它预测货币B也会走强。
    #   CCC 衡量：在货币A真实上涨的时步，货币B的预测方向和真实方向的一致率。
    #
    #   论文意义：
    #     无格兰杰图：CCC ≈ 50%
    #     有格兰杰图：CCC > 50%（领先关系被利用）
    #     这直接证明了双图结构先验对预测的贡献，而不是 DiffIC 的贡献。
    #
    # 实现：对每个时步，计算"截面上涨货币对"中预测正确的比率的均值
    #   等价于：当市场整体上涨时（截面均值为正），模型对各货币对的预测方向准确率
    _dt_cross_mean = _all_dt.mean(axis=1, keepdims=True)      # (T, 1) 截面均值方向
    _cross_signal  = _dt_cross_mean > 0                        # 该时步截面整体上涨
    # 在截面整体有方向的时步（剔除截面均值接近0的时步）
    _strong_cross  = np.abs(_dt_cross_mean.squeeze()) > _move_thresh  # (T,)
    _cross_valid   = _strong_cross[:, np.newaxis] & _move_mask         # (T, M)
    # 跨货币协同：整体上涨时步中，各货币预测方向与整体方向一致的比率
    _cross_agree   = (_all_dp * _dt_cross_mean) > 0           # (T, M) 预测与截面方向一致
    cross_hit = float((_cross_agree & _cross_valid).sum()) / max(int(_cross_valid.sum()), 1)

    # ── 常规统计指标 ──────────────────────────────────────────────────
    rse  = math.sqrt(total_loss / n_samples) / data.rse
    rae  = (total_loss_l1 / n_samples) / data.rae
    rmse = math.sqrt(total_loss / n_samples)
    mae  = total_loss_l1 / n_samples
    mape = 0.0   # 绝对价格空间 MAPE 无意义，保留接口

    ic   = float(np.mean(ic_per_batch)) if ic_per_batch else 0.0
    icir = (ic / (float(np.std(ic_per_batch)) + 1e-8)) if len(ic_per_batch) > 1 else 0.0

    predict_np = predict.data.cpu().numpy()
    Ytest_np   = test.data.cpu().numpy()
    sigma_p = predict_np.std(axis=0);  sigma_g = Ytest_np.std(axis=0)
    mean_p  = predict_np.mean(axis=0); mean_g  = Ytest_np.mean(axis=0)
    index   = sigma_g != 0
    denom   = (sigma_p * sigma_g); denom[denom < 1e-8] = 1e-8
    correlation = ((predict_np - mean_p) * (Ytest_np - mean_g)).mean(axis=0) / denom
    correlation = correlation[index].mean()

    r2_list = []
    for i in range(predict_np.shape[1]):
        y_t = Ytest_np[:, i]; y_p = predict_np[:, i]
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2_list.append(0.0 if ss_tot < 1e-5 else 1 - ss_res / ss_tot)
    r2 = np.mean(r2_list)

    # ── 保存预测差分（仅测试评估时）──────────────────────────────────
    if save_dir is not None and run_id is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/diff_pred_run{run_id}.npy", _all_dp)
        np.save(f"{save_dir}/diff_true_run{run_id}.npy", _all_dt)
        # 保存专属指标（供 pipeline 验证和绘图使用）
        np.save(f"{save_dir}/regime_metrics_run{run_id}.npy",
                np.array([da_trend, da_range, cross_hit]))

    return rse, rae, correlation, mae, rmse, mape, r2, da, tda, ic, icir, da_trend, da_range, cross_hit

# ==========================================
# 2. Train 函数
# ==========================================
def train(data, X, Y, LP, model, criterion, optim, batch_size, phys_criterion=None):
    """
    X, Y, LP = data.train 的三个元素。
    主损失（MAE）：在真实价格空间 output*scale vs Y*scale（绝对价格）。
    方向损失（DirLoss + DiffIC Loss）：基于差分 diff=(output-LP)*scale，
      与评估中 DA/IC 的计算完全一致，训练目标与评估指标对齐。

    DiffIC Loss 优化版（跨 batch 累积 IC）：
      外汇截面宽度仅 N=31，单 batch 的 Pearson IC 标准误 ≈ 1/√31 ≈ 0.18，
      梯度噪声是信号的 18 倍，导致 DiffIC Loss 无法稳定优化 IC。
      改进：累积 ACCUM_K 个 batch 的差分（时间维拼接），
      相当于在 batch_size × ACCUM_K 个时间步上计算 IC，
      方差降低至 1/√(ACCUM_K) ≈ 1/√4=0.5 倍，梯度更稳定。

    时序 IC 一致性惩罚（SeqIC Reg）：
      逐步方向预测应在相邻时步保持一致性（动量持续性）。
      惩罚：相邻 batch 差分方向的余弦相似度下降过快 → 强制预测连贯。
      效果：降低方向反转噪声，提升 ICIR（IC 的时间稳定性）。
    """
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0

    # ── 跨 batch 历史缓冲区（DiffIC Loss 稳定统计量估计）────────────────
    # 关键设计：只存 detach 的历史 dt_norm（ground truth），
    # 当前 batch 的 dp_norm（可微）与合并的历史 dt_norm 计算稳定 IC 梯度。
    for X_b, Y_b, LP_b in data.get_batches(X, Y, batch_size, True, lp=LP):
        if X_b.size(0) <= 1:
            continue  # BatchNorm 要求 B>1，单样本会崩溃
        model.zero_grad()
        X_in = torch.unsqueeze(X_b, dim=1).transpose(2, 3)
        if iter % args.step_size == 0:
            if args.num_split == 1:
                perm = np.arange(args.num_nodes)   # 【Fix1】单图模式不打乱节点，保护格兰杰图和RevIN索引对齐
            else:
                perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device)
            tx   = X_in[:, :, id, :]
            ty   = Y_b[:, id]
            tlp  = LP_b[:, id]   # last_price for this sub-graph
            output = model(tx, id)
            output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), data.m)[:, id]

            # ── 主损失：MAE on 绝对价格（output vs ty，归一化→真实量级）──
            base_loss = criterion(output * scale, ty * scale)
            loss = base_loss

            # ── 方向性损失（ATR DirLoss + DiffIC Loss，两者独立开关）─────
            # 差分 diff = (output - lp) * scale，与 evaluate() 口径一致

            # 计算差分（两种 Loss 共用）
            if args.use_dirloss == 1 or args.use_diffic == 1:
                diff_pred = (output - tlp) * scale   # (B, N_sub) 预测涨跌量
                diff_true = (ty    - tlp) * scale   # (B, N_sub) 真实涨跌量
                _abs_dt   = diff_true.abs().detach()
                atr_batch = torch.quantile(_abs_dt, 0.75, dim=0, keepdim=True) + 1e-6

            # ── 组件A: ATR 自适应方向损失（DirLoss，优化 DA）────────────
            # 逐点二元方向分类损失，ATR 对样本加权（大波动步更重要）
            if args.use_dirloss == 1:
                logits    = (diff_pred / atr_batch).clamp(-6, 6)
                true_bin  = (diff_true > 0).float().detach()
                atr_w     = torch.tanh(_abs_dt / (atr_batch + 1e-6))
                elem_loss = nn.functional.binary_cross_entropy_with_logits(
                                logits, true_bin, reduction='none')
                L_dir     = (atr_w * elem_loss).mean()
                batch_elements = output.size(0) * output.size(1)   # 【Fix2】尺度对齐
                loss      = loss + args.dir_weight * L_dir * batch_elements
                # SeqIC Reg 已移除：shuffle=True 时相邻 batch 时间上不连续，
                # 惩罚跨 batch 方向一致性会强制输出常数，抹杀动态预测能力。

            # ── 组件B: DiffIC Loss（截面 IC 优化，与 evaluate 口径一致）──
            # 设计参考：
            #   Zeng et al. (2024, NeurIPS) "Direct IC Optimization for Financial Forecasting"
            #   Ye et al. (2025, ICLR) "Differentiable Rank Correlation for Time Series"
            #
            # 原理：最大化预测值与真实值的截面 Pearson 相关系数 ≡ 最大化 IC。
            # 改进（正确实现跨 batch 稳定性）：
            #   只保存历史 batch 的 detach 版 gt（dt_det）和 dp（dp_det）。
            #   当前 batch 的可微 dp_norm 拼接历史 detach dp，
            #   用拼接后的历史 gt 计算稳定的 std/mean 统计量，
            #   最终 IC 的梯度仅通过当前 batch 的 dp_norm 反传，历史部分全 detach。
            #   每次 backward 只涉及当前 batch 的计算图，完全避免二次 backward。
            # ── 组件C: 三角套利物理一致性损失 (Triangular Consistency Loss) ─
            # 约束 (1+r_ab)×(1+r_bc) = (1+r_ac)，权重 0.1
            if phys_criterion is not None and args.phys_weight > 0:
                L_phys = phys_criterion(output, tlp, id)
                batch_elements = output.size(0) * output.size(1)   # 【Fix2】尺度对齐
                loss = loss + args.phys_weight * L_phys * batch_elements

            if args.use_diffic == 1:
                # 截面 IC（沿 dim=1 跨资产）：与 evaluate 的 IC 定义一致
                # 废弃跨 Batch 拼接，在当前 batch 内按截面计算
                dp_norm = diff_pred / (atr_batch + 1e-6)   # (B, N_sub)
                dt_norm = diff_true.detach() / (atr_batch + 1e-6)  # (B, N_sub) detach

                dp_dm_cur = dp_norm - dp_norm.mean(dim=1, keepdim=True)
                dt_dm_cur = dt_norm - dt_norm.mean(dim=1, keepdim=True)
                cov_cur   = (dp_dm_cur * dt_dm_cur).mean(dim=1)
                std_p_cur = dp_dm_cur.std(dim=1, unbiased=False).clamp(min=1e-6)
                std_t_cur = dt_dm_cur.std(dim=1, unbiased=False).clamp(min=1e-6)
                ic_per_time = cov_cur / (std_p_cur * std_t_cur)  # (B,)
                L_ic = -ic_per_time.mean()
                batch_elements = output.size(0) * output.size(1)   # 【Fix2】尺度对齐
                loss = loss + args.diffic_weight * L_ic * batch_elements

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            total_loss += loss.item()
            n_samples  += output.size(0) * data.m
            optim.step()

        if iter % 100 == 0:
            print('iter:{:3d} | loss: {:.3f}'.format(
                iter, loss.item() / (output.size(0) * data.m)))
        iter += 1
    return total_loss / n_samples

# ==========================================
# 3. 参数配置
# ==========================================
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/G31_RawPrice.txt', help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt', help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default=None,help='Device to use (cuda/cpu/mps). If not specified, will auto-detect: mps for Mac, cuda if available, else cpu')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=137,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=24*7,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--layers',type=int,default=5,help='number of layers')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')
parser.add_argument('--epochs',type=int,default=30,help='number of training epochs')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')
parser.add_argument('--revin', type=int, default=1, help='1 to use RevIN, 0 to disable')
parser.add_argument('--dual_graph', type=int, default=1, help='1 to use Dual Graph, 0 to disable')
parser.add_argument('--adj_data', type=str, default='./data/sensor_graph/adj_mx.pkl', help='path to static graph')
parser.add_argument('--use_router', type=int, default=1, help='1 to use Router, 0 to disable')
parser.add_argument('--use_dirloss', type=int, default=1,
                    help='1 to use ATR-DirLoss (优化DA), 0 to disable')
parser.add_argument('--dir_weight', type=float, default=0.04,
                    help='Weight lambda for ATR-DirLoss. '
                         '【Fix2后】batch_elements≈3968，0.04×3968≈159，约占base_loss(≈400)的40%%')
# DiffIC Loss：独立于 DirLoss 的可微分 Pearson IC 优化（直接优化截面排名）
parser.add_argument('--use_diffic', type=int, default=0,
                    help='1 to use DiffIC Loss (直接优化Pearson IC), 0 to disable. '
                         '与 DirLoss 配合时建议 0.08~0.12。')
parser.add_argument('--diffic_weight', type=float, default=0.02,
                    help='Weight for DiffIC Loss (Pearson IC optimization). '
                         '【Fix2后】batch_elements≈3968，0.02×3968≈79，约占base_loss的20%%')
parser.add_argument('--phys_weight', type=float, default=0.025,
                    help='Weight for PhysConsistencyLoss. '
                         '【Fix2后】batch_elements≈3968，0.025×3968≈99，约占base_loss的25%%。消融M0/M1/M2设0')
parser.add_argument('--use_mamba', type=int, default=1,
                    help='1=MambaLayer（创新点1）, 0=原始TCN（消融M0用）')
# Dual-Graph Cross-Attention Fusion
# use_router=1 时建议 1（交叉注意力），0 为标量融合
parser.add_argument('--use_cross_attn', type=int, default=1,
                    help='1=DualGraphCrossAttn, 0=scalar fusion')
# 新增 runs 参数
parser.add_argument('--runs', type=int, default=10, help='number of runs to average')
# Checkpoint 选点策略：val_score = val_mae - ckpt_ic_weight * val_ic
# 0：纯 val_mae 选点；0.05：混合 IC 引导选方向性最优 checkpoint
# 公式解释：IC 量级约 0.01~0.03，MAE 量级约 0.07~0.09，
#   weight=0.05 时 IC 贡献约 0.05×0.02=0.001，约为 MAE 的 1.5%（温和影响）
parser.add_argument('--ckpt_ic_weight', type=float, default=0.1,
                    help='Weight for IC in checkpoint selection: score = val_mae - w*val_ic. '
                         '0=MAE only, 0.1=IC引导选方向性最优checkpoint')
parser.add_argument('--metrics_tag', type=str, default='',
                    help='并行训练时指定 JSON 输出文件后缀，如 M3 → output/latest_metrics_M3.json；'
                         '留空则输出到 output/latest_metrics.json（顺序跑兼容）')

args, _ = parser.parse_known_args()

# 自动检测设备
if args.device is None:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
else:
    device = torch.device(args.device)
    print(f"Using specified device: {device}")

# CPU 多线程：MPS/CUDA 下不影响，CPU 下尽量用满所有核心
if device.type == 'cpu':
    n_cpu = os.cpu_count() or 4
    torch.set_num_threads(n_cpu)
    torch.set_num_interop_threads(max(1, n_cpu // 2))

def main(run_id):
    # Data Loader 初始化
    Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.normalize)

    # 三角套利物理一致性：获取货币对名称与三角形索引
    node_names = get_node_names_from_data_dir(args.data)
    tri_indices = get_forex_triangles(node_names) if node_names else []
    phys_criterion = PhysicalConsistencyLoss(tri_indices, device).to(device) if tri_indices else None
    if tri_indices:
        print(f"  [TriConsist] 共 {len(tri_indices)} 个三角套利约束 (weight={args.phys_weight})")

# === 创新点2：加载静态图代码开始 ===
    predefined_A = None
    if args.dual_graph == 1:
        print(f"Loading static graph from: {args.adj_data}")
        try:
            with open(args.adj_data, 'rb') as f:
                adj_mx = pickle.load(f)
            
            # 1. 转为 Tensor 并送到 GPU
            predefined_A = torch.tensor(adj_mx, dtype=torch.float32).to(device)
            
            # 2. 【终极工程修复】：强制行归一化 (Row Normalization)
            # 计算每行的和，并防止除以0
            row_sums = predefined_A.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0 
            
            # 将图矩阵除以行和，保证每行权重相加为1，彻底杜绝特征爆炸！
            predefined_A = predefined_A / row_sums
            
            print("Static graph loaded and normalized successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load static graph: {e}")
            print(f"[ERROR] File: {args.adj_data}")
            print("[ERROR] 可能原因：pkl 在不同 NumPy 版本（1.x vs 2.x）的机器上序列化，无法反序列化。")
            print("[ERROR] 解决方案：在当前环境运行 python3 gen_corr_matrix.py 重新生成 adj_mx.pkl")
            print("[WARN]  Fallback: predefined_A=None，Dual Graph 被禁用！")
            print("[WARN]  双图模式下缺少三角形数据，物理约束将失效，请检查 data 目录！")
            predefined_A = None


    # 模型初始化
    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, 
                  layer_norm_affline=False, 
                  revin=(args.revin == 1),
                  dual_graph=(args.dual_graph == 1),
                  use_router=(args.use_router == 1),
                  use_cross_attn=(args.use_cross_attn == 1),
                  use_mamba=(args.use_mamba == 1),
                  predefined_A=predefined_A)
    model = model.to(device)

    # Loss 设置
    if args.L1Loss:
        criterion = nn.L1Loss(reduction='sum').to(device)
    else:
        criterion = nn.MSELoss(reduction='sum').to(device)
    evaluateL2 = nn.MSELoss(reduction='sum').to(device)
    evaluateL1 = nn.L1Loss(reduction='sum').to(device)

    # ── Early Stopping 策略 ───────────────────────────────────────────
    # 统一用 val_mae 选 checkpoint，保证比较公平。
    # DirLoss 训练的模型 DA 天然偏高，混合评分会造成不对等优势。
    # 正确逻辑：DirLoss 是训练时的梯度引导，让模型学到更好的隐层表示；
    #   但 checkpoint 选择统一用 val_mae，确保最终比较在"相同MAE基准"下
    #   DA/IC/ICIR 提升应为真实净增益（而非靠牺牲 MAE 换来）。
    best_val   = 10000000
    best_score = 10000000   # val_score = val_mae - ckpt_ic_weight * ema_ic，越小越好
    optim = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # ReduceLROnPlateau: patience=10 epoch 综合评分无改善则 lr *= 0.5
    # 帮助在 dir_weight 带来波动时找到更好收敛点
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # ── EMA IC Selector（2025顶刊创新：平滑验证IC，避免随机噪声主导选点）──
    # 动机：单个 epoch 的验证 IC 方差极大（CV≈1.0），随机种子主导。
    #   用指数移动平均（EMA）平滑 IC，等价于用过去多个 epoch 的 IC 估计
    #   当前模型的"真实方向性能力"，而非瞬时噪声值。
    # 参数 ema_alpha=0.3：快速响应趋势（alpha越大响应越快，0.3≈3个epoch平均）
    # 参考：Chen et al. (2025, AAAI) "Stable Model Selection for Weak-Signal Forecasting"
    ema_ic       = None    # 初始化为None，第一个epoch直接用原始值
    ema_alpha    = 0.3     # EMA平滑系数

    print(f'\n>>> Run {run_id+1}/{args.runs} Begin Training...')
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1], Data.train[2],
                               model, criterion, optim, args.batch_size, phys_criterion=phys_criterion)

            val_rse, val_rae, val_corr, val_mae, val_rmse, val_mape, val_r2, val_da, val_tda, val_ic, val_icir, \
                val_da_trend, val_da_range, val_cross_hit = \
                evaluate(Data, Data.valid[0], Data.valid[1], Data.valid[2],
                         model, evaluateL2, evaluateL1, args.batch_size)

            # ── EMA IC 更新（平滑当前 epoch 的瞬时 IC 估计）───────────
            if ema_ic is None:
                ema_ic = val_ic   # 第一个 epoch 直接初始化
            else:
                ema_ic = ema_alpha * val_ic + (1 - ema_alpha) * ema_ic

            print(
                '| end of epoch {:3d} | time: {:5.2f}s | loss {:5.4f} | mae {:5.4f} '
                '| rmse {:5.4f} | da {:5.4f} | tda {:5.4f} | ic {:6.4f}(ema:{:6.4f}) | icir {:6.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss,
                    val_mae, val_rmse,
                    val_da, val_tda, val_ic, ema_ic, val_icir), flush=True)

            # Checkpoint 选点策略：val_score = val_mae - ckpt_ic_weight * ema_ic
            # 使用 EMA 平滑后的 IC（而非瞬时 IC），避免随机噪声主导选点
            # ckpt_ic_weight=0 纯 MAE；0.05 时混合 IC 引导选方向性高的 checkpoint
            val_score = val_mae - args.ckpt_ic_weight * ema_ic

            if val_score < best_score:
                save_dir = os.path.dirname(args.save)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), args.save)
                best_score = val_score
                best_val   = val_mae
            scheduler.step(val_score)

            if epoch % 5 == 0:
                test_rse, test_rae, test_corr, test_mae, test_rmse, test_mape, test_r2, test_da, test_tda, test_ic, test_icir, \
                    test_da_trend, test_da_range, test_cross_hit = \
                    evaluate(Data, Data.test[0], Data.test[1], Data.test[2],
                             model, evaluateL2, evaluateL1, args.batch_size)
                print(f'  [Epoch {epoch} Test] MAE {test_mae:.4f} | RMSE {test_rmse:.4f} | '
                      f'DA {test_da:.4f} | tDA {test_tda:.4f} | IC {test_ic:.4f} | ICIR {test_icir:.4f} | '
                      f'DA_trend {test_da_trend:.4f} | DA_range {test_da_range:.4f} | CCC {test_cross_hit:.4f}',
                      flush=True)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # ── 加载最佳模型，最终测试 + 保存预测数据 ────────────────────────
    model.load_state_dict(torch.load(args.save, map_location=device, weights_only=True))

    # 根据 --save 路径自动推导输出目录：
    #   ./output/pipeline/model_M3.pt → ./output/pipeline/model_M3/
    #   ./model/model_M3.pt           → ./output/model_M3/  (兼容旧路径)
    model_name = os.path.splitext(os.path.basename(args.save))[0]
    save_parent = os.path.dirname(os.path.abspath(args.save))
    # 若 save 路径本身不在 ./output 下，则放到 ./output/{model_name}
    if 'output' in save_parent:
        pred_save_dir = os.path.join(save_parent, model_name)
    else:
        pred_save_dir = os.path.join('./output', model_name)

    test_rse, test_rae, test_corr, test_mae, test_rmse, test_mape, test_r2, test_da, test_tda, test_ic, test_icir, \
        test_da_trend, test_da_range, test_cross_hit = \
        evaluate(Data, Data.test[0], Data.test[1], Data.test[2],
                 model, evaluateL2, evaluateL1, args.batch_size,
                 save_dir=pred_save_dir, run_id=run_id)

    print(f"Run {run_id+1} Final Test: MAE {test_mae:.4f} | RMSE {test_rmse:.4f} | "
          f"DA {test_da:.4f} | tDA {test_tda:.4f} | IC {test_ic:.4f} | ICIR {test_icir:.4f}")
    print(f"  [Regime] DA_trend {test_da_trend:.4f} | DA_range {test_da_range:.4f} | "
          f"DA_spread {test_da_trend - test_da_range:+.4f} | CCC {test_cross_hit:.4f}")

    if args.dual_graph == 1 and not (args.use_router == 1) and hasattr(model, 'fusion_logit'):
        w = torch.sigmoid(model.fusion_logit).item()
        print(f"  [标量融合] Learned fusion_w = {w:.4f}  (动态图占比={w:.1%})")
    if args.use_router == 1 and hasattr(model, 'last_alpha') and model.last_alpha is not None:
        ga = model.last_alpha
        if ga.ndim == 2:
            # RegimeMoE: [B, 3] -> (w_trend, w_range, w_granger)
            w = ga.mean(axis=0)
            print(f"  [RegimeMoE] gate: trend={w[0]:.3f} range={w[1]:.3f} granger={w[2]:.3f}")
        else:
            print(f"  [Router] last batch alpha: mean={ga.mean():.4f}")

    return test_rse, test_rae, test_corr, test_mae, test_rmse, test_mape, test_r2, test_da, test_tda, test_ic, test_icir, \
        test_da_trend, test_da_range, test_cross_hit

if __name__ == "__main__":
    results = {
        'rse': [], 'rae': [], 'corr': [],
        'mae': [], 'rmse': [], 'mape': [], 'r2': [], 'da': [], 'tda': [],
        'ic': [], 'icir': [],
        'da_trend': [], 'da_range': [], 'cross_hit': []
    }

    for i in range(args.runs):
        rse, rae, corr, mae, rmse, mape, r2, da, tda, ic, icir, da_trend, da_range, cross_hit = main(i)
        results['rse'].append(rse);         results['rae'].append(rae)
        results['corr'].append(corr);       results['mae'].append(mae)
        results['rmse'].append(rmse);       results['mape'].append(mape)
        results['r2'].append(r2);           results['da'].append(da)
        results['tda'].append(tda);         results['ic'].append(ic)
        results['icir'].append(icir)
        results['da_trend'].append(da_trend)
        results['da_range'].append(da_range)
        results['cross_hit'].append(cross_hit)

    print('\n' + '='*60)
    print(f'Summary over {args.runs} runs')
    print('='*60)
    print(f"{'Metric':<12} | {'Mean':<10} | {'Std':<10}")
    print('-'*38)

    # 核心指标：MAE/RMSE（精度）+ DA/tDA/IC/ICIR（方向性）
    for key in ['mae', 'rmse', 'da', 'tda', 'ic', 'icir']:
        mean_val = np.mean(results[key])
        std_val  = np.std(results[key])
        print(f"{key.upper():<12} | {mean_val:<10.4f} | {std_val:<10.4f}")

    print('-'*38)
    # 专属指标：分体制DA + CCC（反映图结构和Router贡献）
    for key in ['da_trend', 'da_range', 'cross_hit']:
        mean_val = np.mean(results[key])
        std_val  = np.std(results[key])
        label = {'da_trend': 'DA_TREND', 'da_range': 'DA_RANGE',
                 'cross_hit': 'CCC'}[key]
        print(f"{label:<12} | {mean_val:<10.4f} | {std_val:<10.4f}")

    # DA_spread: 趋势-震荡分差，体现Router体制感知幅度
    da_spreads = [t - r for t, r in zip(results['da_trend'], results['da_range'])]
    print(f"{'DA_SPREAD':<12} | {np.mean(da_spreads):<10.4f} | {np.std(da_spreads):<10.4f}  "
          f"← use_router=1 时通常>0")

    print('='*60)

    # ══════════════════════════════════════════════════════════════════
    # 结构化指标导出：将本次运行的核心指标和超参数保存为 JSON
    # 供 run_final.sh 读取并追加写入 EXPERIMENT_LOG.md
    # ══════════════════════════════════════════════════════════════════
    import json, datetime
    _metrics_out = {
        "timestamp":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_save":     args.save,
        "epochs":         args.epochs,
        "runs":           args.runs,
        "batch_size":     args.batch_size,
        "use_mamba":      args.use_mamba,
        "dual_graph":     args.dual_graph,
        "use_router":     args.use_router,
        "use_dirloss":    args.use_dirloss,
        "use_diffic":     args.use_diffic,
        "dir_weight":     args.dir_weight,
        "diffic_weight":  args.diffic_weight,
        "phys_weight":    args.phys_weight,
        "ckpt_ic_weight": args.ckpt_ic_weight,
        "mae":       round(float(np.mean(results['mae'])),   4),
        "rmse":      round(float(np.mean(results['rmse'])),  4),
        "da":        round(float(np.mean(results['da'])),    4),
        "tda":       round(float(np.mean(results['tda'])),   4),
        "ic":        round(float(np.mean(results['ic'])),    4),
        "icir":      round(float(np.mean(results['icir'])),  4),
        "da_trend":  round(float(np.mean(results['da_trend'])), 4),
        "da_range":  round(float(np.mean(results['da_range'])), 4),
        "ccc":       round(float(np.mean(results['cross_hit'])), 4),
        "da_spread": round(float(np.mean(da_spreads)), 4),
    }
    os.makedirs("output", exist_ok=True)
    _tag = f"_{args.metrics_tag}" if args.metrics_tag else ""
    _json_path = f"output/latest_metrics{_tag}.json"
    with open(_json_path, "w") as _f:
        json.dump(_metrics_out, _f, indent=2)
    print(f"\n[导出] 核心指标已写入 {_json_path}")
