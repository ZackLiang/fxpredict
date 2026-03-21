# -*- coding: utf-8 -*-
"""
plot_bars.py  ── 重新绘制消融/对比柱状图
  Fig 1: metric_ablation_bar.png   消融实验 M0→M3，6个指标子图
  Fig 2: metric_comparison_bar.png Baseline vs 消融 4个指标子图
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family']       = 'DejaVu Sans'
rcParams['axes.spines.top']   = False
rcParams['axes.spines.right'] = False
rcParams['axes.grid']         = True
rcParams['grid.alpha']        = 0.25
rcParams['grid.linestyle']    = '--'

OUTDIR = 'backup_v1_20260302_1053/ppt_figures'
os.makedirs(OUTDIR, exist_ok=True)

# ── 配色 ─────────────────────────────────────────────────────────────────
ABLATION_COLORS = {
    'M0': '#999999', 'M1': '#aec7e8', 'M2': '#1f77b4', 'M3': '#d62728',
}
BASELINE_COLORS = {
    'LSTM': '#9467bd', 'TCN': '#8c564b', 'AGCRN': '#e377c2',
    'PatchTST': '#17becf', 'iTransformer': '#bcbd22',
}
SHORT_LABEL = {
    'M0': 'M0', 'M1': 'M1', 'M2': 'M2', 'M3': 'M3',
    'LSTM': 'LSTM', 'TCN': 'TCN', 'AGCRN': 'AGCRN',
    'PatchTST': 'PatchTST', 'iTransformer': 'iTrans.',
}

# ── V1 真实实验数据（backup_v1_20260302_1053 日志 Summary 块精确解析）──
DATA = {
    # ── Baseline 模型 ────────────────────────────────────────────────────
    'LSTM':         dict(mae=0.2276, mae_std=0.0080, rmse=1.3713, rmse_std=0.0337,
                         da=0.5092,  da_std=0.0012,  tda=0.5142, tda_std=0.0005,
                         ic=0.0141,  ic_std=0.0059,  icir=0.2302, icir_std=0.1094),
    'TCN':          dict(mae=0.1613, mae_std=0.0004, rmse=1.1505, rmse_std=0.0005,
                         da=0.5066,  da_std=0.0008,  tda=0.5069, tda_std=0.0017,
                         ic=0.0052,  ic_std=0.0063,  icir=0.0985, icir_std=0.1129),
    'AGCRN':        dict(mae=0.1752, mae_std=0.0021, rmse=1.1672, rmse_std=0.0011,
                         da=0.5038,  da_std=0.0013,  tda=0.5044, tda_std=0.0015,
                         ic=0.0005,  ic_std=0.0033,  icir=0.0147, icir_std=0.0724),
    'PatchTST':     dict(mae=0.1715, mae_std=0.0008, rmse=1.1787, rmse_std=0.0034,
                         da=0.5118,  da_std=0.0011,  tda=0.5125, tda_std=0.0004,
                         ic=0.0193,  ic_std=0.0034,  icir=0.3627, icir_std=0.0920),
    'iTransformer': dict(mae=0.1717, mae_std=0.0002, rmse=1.1799, rmse_std=0.0028,
                         da=0.5065,  da_std=0.0010,  tda=0.5086, tda_std=0.0024,
                         ic=0.0170,  ic_std=0.0028,  icir=0.3612, icir_std=0.1035),
    # ── 消融实验 M0→M3 ───────────────────────────────────────────────────
    'M0':           dict(mae=0.6442, mae_std=0.0579, rmse=3.3039, rmse_std=0.1845,
                         da=0.4998,  da_std=0.0023,  tda=0.4881, tda_std=0.0045,
                         ic=-0.0073, ic_std=0.0110,  icir=-0.0914, icir_std=0.1378),
    'M1':           dict(mae=0.1625, mae_std=0.0004, rmse=1.1532, rmse_std=0.0016,
                         da=0.5076,  da_std=0.0020,  tda=0.5127, tda_std=0.0040,
                         ic=0.0210,  ic_std=0.0122,  icir=0.3953, icir_std=0.2053),
    'M2':           dict(mae=0.1627, mae_std=0.0004, rmse=1.1539, rmse_std=0.0018,
                         da=0.5080,  da_std=0.0007,  tda=0.5115, tda_std=0.0025,
                         ic=0.0261,  ic_std=0.0067,  icir=0.4607, icir_std=0.1162),
    'M3':           dict(mae=0.1629, mae_std=0.0003, rmse=1.1533, rmse_std=0.0005,
                         da=0.5102,  da_std=0.0018,  tda=0.5146, tda_std=0.0030,
                         ic=0.0226,  ic_std=0.0036,  icir=0.3911, icir_std=0.0330),
}


def _ylim_tight(vals, lower_is_better=False, top_pad=0.38, bot_pad=0.15):
    """
    根据数据自动计算紧凑的 Y 轴范围：
    - 下边界贴近最小值（留 bot_pad 倍极差）
    - 上边界留 top_pad 倍极差（给数值标注留空间）
    """
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng < 1e-8:
        rng = abs(hi) * 0.1 if abs(hi) > 1e-8 else 0.01
    ymin = lo - rng * bot_pad
    ymax = hi + rng * top_pad
    return ymin, ymax


def _annotate_bars(ax, bars, vals, stds, fontsize=9):
    """在每个 bar 顶端标注数值"""
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.015
    for bar, v, s in zip(bars, vals, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + offset,
            f'{v:.4f}',
            ha='center', va='bottom',
            fontsize=fontsize, fontweight='bold',
        )


# ══════════════════════════════════════════════════════════════════════════
#  Fig 1: 消融实验指标柱状图  M0→M3  (2×3 子图)
# ══════════════════════════════════════════════════════════════════════════
def plot_ablation_bar(data=DATA, dpi=150):
    ablation_keys = ['M0', 'M1', 'M2', 'M3']
    metrics      = ['mae',  'rmse',  'da',   'tda',  'ic',  'icir']
    metric_label = ['MAE ↓','RMSE ↓','DA ↑','tDA ↑','IC ↑','ICIR ↑']
    lower_better = [True,    True,    False,  False,  False, False]

    # 2行3列，figsize 固定为 PPT 友好尺寸
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Ablation Study — Metric Comparison  (M0 → M3)',
                 fontsize=14, fontweight='bold', y=0.98)
    axes = axes.flatten()

    for ax, metric, mlabel, lb in zip(axes, metrics, metric_label, lower_better):
        keys   = [k for k in ablation_keys if k in data and metric in data[k]]
        vals   = [data[k][metric]                      for k in keys]
        stds   = [data[k].get(metric + '_std', 0.0)   for k in keys]
        colors = [ABLATION_COLORS[k]                   for k in keys]
        xlbls  = [SHORT_LABEL[k]                       for k in keys]
        x      = np.arange(len(keys))

        # ── 绘制柱子 + 误差棒 ──────────────────────────────────────────
        bars = ax.bar(x, vals, color=colors, width=0.6, alpha=0.88,
                      edgecolor='white', linewidth=0.8)
        ax.errorbar(x, vals, yerr=stds, fmt='none',
                    ecolor='#333333', elinewidth=1.2, capsize=4, capthick=1.2)

        # ── Y 轴范围（先设，再标注，避免 offset 计算错误）──────────────
        ymin, ymax = _ylim_tight(vals, lower_is_better=lb)
        ax.set_ylim(ymin, ymax)

        # ── 数值标注 ──────────────────────────────────────────────────
        _annotate_bars(ax, bars, vals, stds, fontsize=8.5)

        # ── 高亮最优 bar ──────────────────────────────────────────────
        if lb:
            # MAE/RMSE 排除 M0（量级不同）
            cmp = vals[1:] if metric in ('mae', 'rmse') and len(vals) > 1 else vals
            off = 1       if metric in ('mae', 'rmse') and len(vals) > 1 else 0
            best = int(np.argmin(cmp)) + off
        else:
            best = int(np.argmax(vals))
        bars[best].set_edgecolor('#d62728')
        bars[best].set_linewidth(2.5)

        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=10)
        ax.set_title(mlabel, fontsize=12, pad=6)
        ax.tick_params(axis='y', labelsize=8)

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.92, wspace=0.35, hspace=0.45)
    path = os.path.join(OUTDIR, 'metric_ablation_bar.png')
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f'[Fig 1] 已保存 → {path}')


# ══════════════════════════════════════════════════════════════════════════
#  Fig 2: Baseline vs 消融 综合对比图  (1×4 子图)
# ══════════════════════════════════════════════════════════════════════════
def plot_comparison_bar(data=DATA, dpi=150):
    metrics      = ['mae',   'da',   'tda',  'ic']
    metric_label = ['MAE ↓', 'DA ↑', 'tDA ↑','IC ↑']
    lower_better = [True,    False,  False,   False]

    baseline_keys = ['LSTM', 'TCN', 'AGCRN', 'PatchTST', 'iTransformer']
    ablation_keys = ['M0',   'M1',  'M2',    'M3']

    # 2行2列布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Baseline Comparison + Ablation Study',
                 fontsize=14, fontweight='bold', y=0.98)
    axes = axes.flatten()

    for ax, metric, mlabel, lb in zip(axes, metrics, metric_label, lower_better):
        # ── 收集数据（Baseline | 空隙 | Ablation）─────────────────────
        grp_keys   = []
        grp_vals   = []
        grp_stds   = []
        grp_colors = []
        grp_labels = []
        grp_edges  = []
        grp_lw     = []

        for k in baseline_keys:
            if k in data and metric in data[k]:
                grp_keys.append(k)
                grp_vals.append(data[k][metric])
                grp_stds.append(data[k].get(metric + '_std', 0.0))
                grp_colors.append(BASELINE_COLORS[k])
                grp_labels.append(SHORT_LABEL[k])
                grp_edges.append('white')
                grp_lw.append(0.8)

        # 空隙占位（不画 bar，仅占 x 位置）
        gap_pos = len(grp_keys)

        for k in ablation_keys:
            if k in data and metric in data[k]:
                grp_keys.append(k)
                grp_vals.append(data[k][metric])
                grp_stds.append(data[k].get(metric + '_std', 0.0))
                grp_colors.append(ABLATION_COLORS[k])
                grp_labels.append(SHORT_LABEL[k])
                grp_edges.append('#d62728' if k == 'M3' else 'white')
                grp_lw.append(2.5 if k == 'M3' else 0.8)

        # x 位置：baseline 连续排，留 1.5 个单位间隙，ablation 接着排
        n_bl = gap_pos
        n_ab = len(grp_keys) - gap_pos
        x_bl = np.arange(n_bl, dtype=float)
        x_ab = np.arange(n_bl + 1.5, n_bl + 1.5 + n_ab, dtype=float)
        x = np.concatenate([x_bl, x_ab])

        # ── Y 轴范围先确定（包含负值，只排除分隔符0值）────────────────
        valid = [v for v, k in zip(grp_vals, grp_keys) if k is not None]
        ymin, ymax = _ylim_tight(valid, lower_is_better=lb, top_pad=0.42, bot_pad=0.15)
        ax.set_ylim(ymin, ymax)

        # ── 画 bar ────────────────────────────────────────────────────
        bars = ax.bar(x, grp_vals, color=grp_colors, width=0.6, alpha=0.88,
                      edgecolor=grp_edges, linewidth=grp_lw)

        # ── 误差棒 ────────────────────────────────────────────────────
        ax.errorbar(x, grp_vals, yerr=grp_stds, fmt='none',
                    ecolor='#444444', elinewidth=1.1, capsize=3, capthick=1.1)

        # ── 数值标注（仅标 M3） ────────────────────────────────────────
        yspan = ymax - ymin
        for xi, k, v in zip(x, grp_keys, grp_vals):
            if k == 'M3':
                ax.text(xi, v + yspan * 0.016,
                        f'{v:.4f}', ha='center', va='bottom',
                        fontsize=8.5, fontweight='bold', color='#d62728')

        # ── 分隔线 + 标签 ──────────────────────────────────────────────
        sep = (x[gap_pos - 1] + x[gap_pos]) / 2
        ax.axvline(sep, color='#888888', lw=1.0, linestyle=':')
        bl_center = np.mean(x[:gap_pos])
        ab_center = np.mean(x[gap_pos:])
        ax.text(bl_center, ymax - yspan * 0.04,
                'Baseline', ha='center', fontsize=8, color='#555555')
        ax.text(ab_center, ymax - yspan * 0.04,
                'Ablation', ha='center', fontsize=8, color='#555555')

        # ── X 轴标签 ──────────────────────────────────────────────────
        ax.set_xticks(x)
        ax.set_xticklabels(grp_labels, fontsize=8, rotation=30, ha='right')
        ax.set_title(mlabel, fontsize=12, pad=8)
        ax.tick_params(axis='y', labelsize=8)

    # 手动调整留白：bottom=0.12 给旋转标签空间，top=0.93 给 suptitle
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12, top=0.93, wspace=0.30, hspace=0.50)
    path = os.path.join(OUTDIR, 'metric_comparison_bar.png')
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f'[Fig 2] 已保存 → {path}')


def print_tables():
    """打印消融表格和Baseline对比表格（含均值±标准差）"""
    metrics_show  = ['mae',   'rmse',  'da',    'tda',   'ic',    'icir']
    headers_show  = ['MAE↓',  'RMSE↓', 'DA↑',   'tDA↑',  'IC↑',   'ICIR↑']
    col_w = 18

    # ── 消融实验表格 ─────────────────────────────────────────────────────
    ablation_keys = ['M0', 'M1', 'M2', 'M3']
    print('\n' + '='*90)
    print('  Ablation Study  (V1 真实数据，3-run mean ± std)')
    print('='*90)
    header = f"{'Model':<10}" + ''.join(f"{h:>{col_w}}" for h in headers_show)
    print(header)
    print('-'*90)
    for k in ablation_keys:
        d = DATA[k]
        row = f"{k:<10}"
        for m in metrics_show:
            v, s = d.get(m, float('nan')), d.get(m+'_std', 0.0)
            cell = f"{v:.4f}±{s:.4f}"
            row += f"{cell:>{col_w}}"
        print(row)
    print('='*90)

    # ── Baseline 对比表格 ────────────────────────────────────────────────
    baseline_keys = ['LSTM', 'TCN', 'AGCRN', 'PatchTST', 'iTransformer',
                     'M3']   # 最后加 M3 对比
    print('\n' + '='*90)
    print('  Baseline Comparison  (V1 真实数据，3-run mean ± std，末行为 Ours M3)')
    print('='*90)
    print(header)
    print('-'*90)
    for k in baseline_keys:
        d = DATA[k]
        row = f"{k:<10}"
        for m in metrics_show:
            v, s = d.get(m, float('nan')), d.get(m+'_std', 0.0)
            cell = f"{v:.4f}±{s:.4f}"
            row += f"{cell:>{col_w}}"
        print(row)
    print('='*90)


# ══════════════════════════════════════════════════════════════════════════
#  Fig 3: Baseline 价格走势拟合对比图  (2×2，风格与 price_fit_comparison 一致)
# ══════════════════════════════════════════════════════════════════════════
# 31 个货币对标签（与原始数据列顺序一致）
PAIR_NAMES = [
    'AUD/USD', 'GBP/USD', 'USD/JPY', 'EUR/USD', 'NZD/USD',
    'USD/MXN', 'CHF/USD', 'USD/CAD', 'EUR/JPY', 'GBP/JPY',
    'EUR/GBP', 'EUR/CHF', 'AUD/JPY', 'GBP/CHF', 'USD/HKD',
    'USD/SGD', 'EUR/CAD', 'GBP/CAD', 'EUR/NZD', 'GBP/JPY-2',
    'EUR/AUD', 'AUD/CHF', 'NZD/USD-2', 'NZD/JPY', 'NZD/CHF',
    'USD/CHF', 'AUD/CAD', 'CAD/JPY', 'USD/CNH', 'USD/CNY',
    'USD/KRW',
]

# Baseline 配色（与 plot_bars 里保持一致）
BASELINE_LINE_STYLE = {
    'LSTM':         ('#9467bd', 1.4, '--'),
    'TCN':          ('#8c564b', 1.4, '-.'),
    'AGCRN':        ('#e377c2', 1.4, ':'),
    'PatchTST':     ('#17becf', 1.8, '-'),
    'iTransformer': ('#bcbd22', 1.8, (0,(3,1,1,1))),
}


def _load_preds(base_dir: str, model_keys: list,
                data_path: str = 'data/G31_RawPrice.txt',
                n_runs: int = 3):
    """加载原始价格 + 多个模型的预测差分，还原成绝对价格序列。"""
    rawdata    = np.loadtxt(data_path, delimiter=',')   # (T, 31)
    N          = rawdata.shape[0]
    val_end    = int(N * 0.8)
    true_prices = rawdata[val_end:, :]                  # (T_test, 31)

    results = {}
    for key in model_keys:
        out_dir = os.path.join(base_dir, f'model_{key}')
        preds_list, trues_list = [], []
        for i in range(n_runs):
            fp = os.path.join(out_dir, f'diff_pred_run{i}.npy')
            ft = os.path.join(out_dir, f'diff_true_run{i}.npy')
            if os.path.exists(fp) and os.path.exists(ft):
                preds_list.append(np.load(fp))
                trues_list.append(np.load(ft))
        if not preds_list:
            print(f'  [SKIP] {key}: no pred files in {out_dir}')
            continue
        min_len = min(p.shape[0] for p in preds_list)
        dp = np.mean([p[:min_len] for p in preds_list], axis=0)
        dt = np.mean([t[:min_len] for t in trues_list], axis=0)

        T = min(len(dp), len(true_prices))
        anchor         = rawdata[val_end - 1, :]
        true_price_seq = true_prices[:T]
        last_price_seq = np.vstack([anchor[np.newaxis, :], true_prices[:T-1]])
        pred_price_seq = last_price_seq + dp[:T]

        results[key] = dict(
            pred_price=pred_price_seq,
            true_price=true_price_seq,
            dp=dp[:T],
            dt=dt[:T],
        )
    return true_prices, results


def plot_baseline_price_fit(
        base_dir: str = 'backup_v1_20260302_1053/output',
        data_path: str = 'data/G31_RawPrice.txt',
        window: int = 400,
        offset: int = 1000,
        n_runs: int = 3,
        dpi: int = 150):
    """
    Baseline 价格走势拟合对比图（2×2，4 个货币对）。
    每个子图：真实价格（蓝）+ 5 条 Baseline 预测线 + M3（红粗线）作参照。
    右下角文字框显示各 Baseline 的 DA / IC。
    """
    # 4 个代表性货币对：USD/JPY(col2) EUR/JPY(col8) USD/KRW(col30) USD/MXN(col5)
    COLS      = [2, 8, 30, 5]
    COL_NAMES = [PAIR_NAMES[c] for c in COLS]

    baseline_keys = ['LSTM', 'TCN', 'AGCRN', 'PatchTST', 'iTransformer']
    all_keys      = baseline_keys + ['M3']

    true_prices, model_results = _load_preds(
        base_dir=base_dir,
        model_keys=all_keys,
        data_path=data_path,
        n_runs=n_runs,
    )

    if not model_results:
        print('[SKIP] baseline_price_fit: 未找到任何预测文件')
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        'Price Trend Fitting — Baselines vs M3 (Ours)\n'
        f'Test Set  |  Window: {window} hrs  |  3-run Average',
        fontsize=13, fontweight='bold', y=0.98,
    )
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    for ax, col_idx, pair_name in zip(axes, COLS, COL_NAMES):
        t_start = offset
        t_end   = min(offset + window, true_prices.shape[0])
        t_range = np.arange(t_start, t_end)

        # ── Ground Truth ──────────────────────────────────────────────
        true_seg = true_prices[t_start:t_end, col_idx]
        ax.plot(t_range, true_seg,
                color='#1f77b4', lw=2.2, zorder=6,
                label='Ground Truth', alpha=0.95)

        # ── M3 参照线（红色粗实线，放在最上层） ────────────────────────
        metric_lines = []
        if 'M3' in model_results:
            pred_seg = model_results['M3']['pred_price'][t_start:t_end, col_idx]
            ax.plot(t_range, pred_seg,
                    color='#d62728', lw=2.2, linestyle='-',
                    zorder=7, alpha=0.90, label='M3 (Ours)')
            dp_s = model_results['M3']['dp'][t_start:t_end, col_idx]
            dt_s = model_results['M3']['dt'][t_start:t_end, col_idx]
            mv   = np.abs(dt_s) > 1e-6
            da_  = (dp_s * dt_s > 0)[mv].mean() if mv.sum() > 0 else 0.5
            ic_  = ((dp_s - dp_s.mean()) * (dt_s - dt_s.mean())).mean() / \
                   (dp_s.std() * dt_s.std() + 1e-9)
            metric_lines.append(f'M3 (Ours): DA={da_:.3f} IC={ic_:.3f}')

        # ── 5 条 Baseline 预测线 ───────────────────────────────────────
        for zord, bkey in enumerate(baseline_keys, start=1):
            if bkey not in model_results:
                continue
            color, lw, ls = BASELINE_LINE_STYLE[bkey]
            pred_seg = model_results[bkey]['pred_price'][t_start:t_end, col_idx]
            ax.plot(t_range, pred_seg,
                    color=color, lw=lw, linestyle=ls,
                    zorder=zord, alpha=0.80,
                    label=bkey)

            dp_s = model_results[bkey]['dp'][t_start:t_end, col_idx]
            dt_s = model_results[bkey]['dt'][t_start:t_end, col_idx]
            mv   = np.abs(dt_s) > 1e-6
            da_  = (dp_s * dt_s > 0)[mv].mean() if mv.sum() > 0 else 0.5
            ic_  = ((dp_s - dp_s.mean()) * (dt_s - dt_s.mean())).mean() / \
                   (dp_s.std() * dt_s.std() + 1e-9)
            metric_lines.append(f'{bkey:13s}: DA={da_:.3f} IC={ic_:.3f}')

        # ── 指标文字框 ────────────────────────────────────────────────
        ax.text(0.98, 0.04, '\n'.join(metric_lines),
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7.5, family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.88, edgecolor='#cccccc'))

        ax.set_title(f'{pair_name}', fontsize=11, pad=6)
        ax.set_xlabel('Test Set Time Step (hours)', fontsize=9)
        ax.set_ylabel('Price', fontsize=9)
        ax.legend(loc='upper left', fontsize=7.5, framealpha=0.85,
                  ncol=2, columnspacing=0.8, handlelength=1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08,
                        top=0.91, wspace=0.28, hspace=0.42)
    path = os.path.join(OUTDIR, 'price_fit_baselines.png')
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f'[Fig 3] 已保存 → {path}')


if __name__ == '__main__':
    print_tables()
    plot_ablation_bar()
    plot_comparison_bar()
    plot_baseline_price_fit()
    from PIL import Image
    for f in ['metric_ablation_bar.png', 'metric_comparison_bar.png',
              'price_fit_baselines.png']:
        p = os.path.join(OUTDIR, f)
        if os.path.exists(p):
            img = Image.open(p)
            print(f'\n  {f}: {img.size[0]}x{img.size[1]} px  → {OUTDIR}/')
