# -*- coding: utf-8 -*-
"""
backtest.py
===========
量化回测脚本（论文最终版）：基于模型保存的 diff_pred / diff_true numpy 数组，
在相同交易规则和手续费假设下，评估各模型的经济学绩效。

策略：Vol-Confidence Filtering（置信度阈值过滤）
  - 设计初衷：验证 DirLoss 和 Regime Router 的方向捕捉能力
  - 逻辑：仅当预测绝对值超过该品种历史 q 分位阈值时入场，规避微观噪音
  - 收益单位：vol-normalised σ units（无量纲，消除不同货币对量纲差异）
  - 手续费：每次换手扣除 cost_ratio × σ（与收益单位对齐）

输出：
  backtest_baselines.png   论文图1：Proposed vs SOTA Baselines
  backtest_ablations.png   论文图2：Ablation Study
  backtest_metrics.json    结构化指标汇总

用法：
  conda run -n fxpredict python3 backtest.py
  conda run -n fxpredict python3 backtest.py --cost_bps 0.002 --vol_quantile 0.75
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ══════════════════════════════════════════════════════════════════════
# 0. 全局配置
# ══════════════════════════════════════════════════════════════════════
# 小时频数据：252 交易日 × 24 小时
ANNUAL_STEPS = 252 * 24

# 所有参与回测的模型（目录名 → 显示名）
MODELS = {
    'model_M3_Proposed':  'M3-Full (Ours)',
    'model_M2':           'M2-DualGraph',
    'model_M1':           'M1-Regime',
    'model_M0':           'M0-Mamba',
    'model_tcn':          'TCN',
    'model_lstm':         'LSTM',
    'model_agcrn':        'AGCRN',
    'model_itransformer': 'iTransformer',
}

# 图1：Baseline 对比（Proposed vs SOTA）
BASELINES_GROUP = [
    'model_M3_Proposed',
    'model_itransformer',
    'model_agcrn',
    'model_tcn',
    'model_lstm',
]

# 图2：消融实验（M0→M3 逐级递进）
ABLATION_GROUP = [
    'model_M3_Proposed',
    'model_M2',
    'model_M1',
    'model_M0',
]

# ── 配色：M3 深红主角，其余陪衬 ──────────────────────────────────────
COLORS = {
    'model_M3_Proposed':  'crimson',
    'model_M2':           '#8E44AD',
    'model_M1':           '#2471A3',
    'model_M0':           '#5D6D7E',
    'model_tcn':          '#1E8449',
    'model_lstm':         '#E67E22',
    'model_agcrn':        '#7F8C8D',
    'model_itransformer': '#AAB7B8',
}

# M3 实线加粗，其余细线带透明度
LW_MAIN   = 2.5   # M3 线宽
LW_OTHER  = 1.2   # 其余线宽
ALPHA_OTHER = 0.8 # 其余透明度


# ══════════════════════════════════════════════════════════════════════
# 1. 数据加载：对 3 runs 平均预测信号
# ══════════════════════════════════════════════════════════════════════
def load_data(model_dir, n_runs=3, base='output'):
    """加载 n_runs 次预测差分，取均值作为最终信号。"""
    preds, trues = [], []
    for r in range(n_runs):
        fp = os.path.join(base, model_dir, f'diff_pred_run{r}.npy')
        ft = os.path.join(base, model_dir, f'diff_true_run{r}.npy')
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        preds.append(np.load(fp))
        trues.append(np.load(ft))
    pred_avg = np.mean(np.stack(preds), axis=0)  # (T, M)
    true_avg = np.mean(np.stack(trues), axis=0)  # (T, M)
    return pred_avg, true_avg


# ══════════════════════════════════════════════════════════════════════
# 2. 手续费：Turnover-aware，与收益单位完全对齐
# ══════════════════════════════════════════════════════════════════════
def turnover_cost(pos, cost_ratio):
    """
    pos        : (T, M) 每步仓位
    cost_ratio : float  每次换手扣除的比例（收益已 vol-normalised，此处同单位）

    收益已归一化（除以 per-instrument σ），手续费同样在归一化空间扣除：
      扣费 = |仓位变动| × cost_ratio
    无论货币对绝对量级如何，手续费均为波动率的固定比例，公平且无量纲。
    """
    delta = np.abs(np.diff(pos, axis=0, prepend=0))  # (T, M)
    return (delta * cost_ratio).sum(axis=1)           # (T,)


# ══════════════════════════════════════════════════════════════════════
# 3. 绩效指标
# ══════════════════════════════════════════════════════════════════════
def calc_metrics(pnl):
    """
    pnl : (T,) 每步净损益（vol-normalised σ 单位）
    返回 dict：total / ann_ret / sharpe / max_dd / calmar
    注：Sharpe / max_dd / calmar 为无量纲，可直接用于论文对比
    """
    cum     = np.cumsum(pnl)
    total   = float(cum[-1])
    n_years = len(pnl) / ANNUAL_STEPS
    ann_ret = total / n_years if n_years > 0 else 0.0
    sharpe  = float(pnl.mean() / (pnl.std(ddof=1) + 1e-12) * np.sqrt(ANNUAL_STEPS))
    running = np.maximum.accumulate(cum)
    max_dd  = float((cum - running).min())
    calmar  = ann_ret / (abs(max_dd) + 1e-12)
    return dict(total=total, ann_ret=ann_ret, sharpe=sharpe,
                max_dd=max_dd, calmar=calmar)


# ══════════════════════════════════════════════════════════════════════
# 4. 策略：自校准置信度过滤（Self-Calibrated Quantile Threshold）
# ══════════════════════════════════════════════════════════════════════
def strategy_vol_conf(pred, true, q=0.75, cost_ratio=0.002):
    """
    Self-Calibrated Quantile Threshold（自校准分位阈值）

    阈值基于模型自身预测分布计算，而非外部真实波动率：
      threshold[i] = quantile(|pred[:, i]|, q)   ← 每品种独立

    学术意义：
      强制每个模型只交易自己"最有把握的 Top (1-q) 时间步"。
      无论架构输出的绝对量级（M3 保守 vs iTransformer 激进），
      所有模型的活跃步比例在数学上收敛到同一水平（验证：active%=25%）。
      彻底消除"预测方差大 → 白嫖更多交易频次"的系统性偏差
      （Uncalibrated Confidence Trap）。

    收益单位：
      ret_norm[t,i] = diff_true[t,i] / σ_i  →  "几个标准差"（无量纲）
    """
    # per-instrument 自校准阈值（来自模型自身预测分布）
    thr       = np.quantile(np.abs(pred), q, axis=0)    # (M,)
    high_conf = np.abs(pred) > thr[np.newaxis, :]        # (T, M) bool
    raw       = np.where(high_conf, np.sign(pred), 0.0)
    n_act     = (raw != 0).sum(axis=1, keepdims=True).clip(min=1)
    pos       = (raw / n_act).astype(np.float32)         # (T, M)

    # vol-normalised 收益
    true_std  = true.std(axis=0).clip(min=1e-8)          # (M,)
    ret_norm  = true / true_std[np.newaxis, :]            # (T, M)
    gross     = (pos * ret_norm).sum(axis=1)              # (T,)

    # 手续费（同单位）
    cost  = turnover_cost(pos, cost_ratio)
    net   = gross - cost

    avg_to = np.abs(np.diff(pos, axis=0, prepend=0)).sum(axis=1).mean()
    return gross, net, pos, avg_to


# ══════════════════════════════════════════════════════════════════════
# 5. 主回测循环
# ══════════════════════════════════════════════════════════════════════
def run_all(args):
    all_results = {}

    print(f'\n{"="*72}')
    print(f'  Vol-Confidence Strategy | q={args.vol_quantile} | '
          f'cost={args.cost_bps}σ/trade | runs={args.runs}')
    print(f'{"="*72}')
    print(f'  {"Model":<26} {"Sharpe":>8} {"CumPnL(σ)":>11} '
          f'{"MaxDD":>9} {"Calmar":>8} {"Turnover":>10}')
    print(f'  {"-"*72}')

    for model_dir, disp in MODELS.items():
        dpath = os.path.join(args.base, model_dir)
        if not os.path.isdir(dpath):
            continue
        try:
            pred, true = load_data(model_dir, n_runs=args.runs, base=args.base)
        except FileNotFoundError as e:
            print(f'  [SKIP] {e}')
            continue

        gross, net, pos, avg_to = strategy_vol_conf(
            pred, true, q=args.vol_quantile, cost_ratio=args.cost_bps)

        gm = calc_metrics(gross)
        nm = calc_metrics(net)

        all_results[model_dir] = {
            'display':      disp,
            'gross_pnl':    gross,
            'net_pnl':      net,
            'gross':        gm,
            'net':          nm,
            'avg_turnover': float(avg_to),
        }

        print(f'  {disp:<26} {nm["sharpe"]:>+8.3f} {nm["total"]:>+11.2f} '
              f'{nm["max_dd"]:>+9.4f} {nm["calmar"]:>+8.2f} {avg_to:>10.4f}')

    print(f'{"="*72}\n')
    return all_results


# ══════════════════════════════════════════════════════════════════════
# 6. 通用绘图函数（论文级，上：累计收益，下：回撤）
# ══════════════════════════════════════════════════════════════════════
def _set_rcparams():
    plt.rcParams.update({
        'font.family':        'DejaVu Sans',
        'font.size':          11,
        'axes.linewidth':     0.9,
        'grid.alpha':         0.35,
        'grid.linestyle':     '--',
        'legend.framealpha':  0.95,
        'legend.edgecolor':   '#CCCCCC',
        'figure.facecolor':   'white',
        'axes.facecolor':     '#F8F8F8',
    })


def plot_group(all_results, model_group, title, out_path, args):
    """
    通用论文级绘图：
      上图 — 累计净收益曲线（Cumulative Risk-Adjusted PnL）
      下图 — 回撤曲线
      右侧 inset — 所有参与模型的 Sharpe 柱状图

    参数：
      all_results  : run_all() 返回的完整结果字典
      model_group  : 本图参与的模型 key 列表（有序）
      title        : 图表标题
      out_path     : 输出文件路径
    """
    _set_rcparams()

    # 过滤掉未跑出结果的模型
    group = [m for m in model_group if m in all_results]
    if not group:
        print(f'  [WARN] 无可用结果，跳过 {out_path}')
        return

    T_total = max(len(all_results[m]['net_pnl']) for m in group)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8.5),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.06},
        sharex=True)

    for model_dir in group:
        r    = all_results[model_dir]
        net  = r['net_pnl']
        cum  = np.cumsum(net)
        xs   = np.arange(len(cum))
        nm   = r['net']
        disp = r['display']
        col  = COLORS.get(model_dir, '#888888')

        is_main = (model_dir == 'model_M3_Proposed')
        lw      = LW_MAIN if is_main else LW_OTHER
        alpha   = 1.0     if is_main else ALPHA_OTHER
        zorder  = 10      if is_main else 2
        ls      = '-'     # 统一实线，用颜色和粗细区分

        # 图例：显示名 + Sharpe + 最终 PnL
        label = (f"{disp}"
                 f"  (Sharpe: {nm['sharpe']:+.2f},"
                 f"  PnL: {nm['total']:+.1f}σ,"
                 f"  Calmar: {nm['calmar']:+.2f})")

        # ── 累计收益 ──
        ax1.plot(xs, cum,
                 color=col, lw=lw, ls=ls, alpha=alpha,
                 label=label, zorder=zorder)

        # ── 回撤 ──
        running = np.maximum.accumulate(cum)
        dd      = cum - running
        ax2.fill_between(xs, dd, 0, color=col, alpha=0.18 if not is_main else 0.30,
                         zorder=zorder)
        ax2.plot(xs, dd, color=col, lw=lw * 0.6, ls=ls, alpha=alpha, zorder=zorder)

    # ── ax1 装饰 ────────────────────────────────────────────────────
    ax1.axhline(0, color='black', lw=0.7, ls='--', alpha=0.45)
    ax1.set_ylabel('Cumulative Risk-Adjusted PnL  (σ units)', fontsize=11)
    ax1.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax1.legend(loc='upper left', fontsize=9.5, ncol=1,
               handlelength=2.5, labelspacing=0.4)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax1.grid(True)

    # ── ax2 装饰 ────────────────────────────────────────────────────
    ax2.set_xlabel('Time Step (hourly bar)', fontsize=11)
    ax2.set_ylabel('Drawdown  (σ units)', fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    xticks = np.arange(0, T_total + 1, 1000)
    ax2.set_xticks(xticks)
    ax2.grid(True)

    # ── 右侧 inset：本组所有模型的 Sharpe 柱状图 ────────────────────
    ax_bar  = fig.add_axes([0.735, 0.575, 0.195, 0.295])
    bars_v  = [all_results[m]['net']['sharpe'] for m in group]
    bar_c   = [COLORS.get(m, '#AAAAAA') for m in group]
    bars_y  = np.arange(len(group))
    bar_h   = ax_bar.barh(bars_y, bars_v, color=bar_c,
                           height=0.55, edgecolor='white', lw=0.5)
    # M3 柱子边框加粗
    if 'model_M3_Proposed' in group:
        idx_m3 = group.index('model_M3_Proposed')
        bar_h[idx_m3].set_edgecolor('crimson')
        bar_h[idx_m3].set_linewidth(1.8)
    ax_bar.set_yticks(bars_y)
    ax_bar.set_yticklabels([all_results[m]['display'] for m in group], fontsize=7.5)
    ax_bar.axvline(0, color='black', lw=0.7)
    ax_bar.set_xlabel('Net Sharpe Ratio', fontsize=8)
    ax_bar.set_title('Sharpe Summary', fontsize=8, fontweight='bold')
    ax_bar.tick_params(labelsize=7)
    ax_bar.grid(axis='x', alpha=0.3, ls='--')

    # ── 页脚注释 ────────────────────────────────────────────────────
    fig.text(0.01, 0.005,
             f'Out-of-sample test | {T_total} hourly steps | '
             f'Self-Calibrated q={args.vol_quantile} | '
             f'Cost = {args.cost_bps}×σ/turnover (vol-normalised) | '
             f'Runs averaged = {args.runs}',
             fontsize=8, color='#888888')

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  [图表] 已保存: {out_path}')


# ══════════════════════════════════════════════════════════════════════
# 7. 保存 JSON 指标
# ══════════════════════════════════════════════════════════════════════
def save_json(all_results):
    out = {}
    for k, v in all_results.items():
        nm = v['net']
        gm = v['gross']
        out[k] = {
            'display': v['display'],
            # NOTE: 所有收益指标单位为「vol-normalised σ units」，非百分比
            # Sharpe / max_dd / calmar 为无量纲，可直接用于论文对比
            # cum_pnl 单位是「σ 之和」，仅供内部参考
            'gross': {
                'sharpe':  round(gm['sharpe'],  4),
                'max_dd':  round(gm['max_dd'],  4),
                'calmar':  round(gm['calmar'],  4),
                'cum_pnl': round(gm['total'],   4),
            },
            'net': {
                'sharpe':  round(nm['sharpe'],  4),
                'max_dd':  round(nm['max_dd'],  4),
                'calmar':  round(nm['calmar'],  4),
                'cum_pnl': round(nm['total'],   4),
            },
            'avg_turnover': round(v['avg_turnover'], 6),
        }
    path = 'backtest_metrics.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  [指标] 已保存: {path}')


# ══════════════════════════════════════════════════════════════════════
# 8. 入口
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Vol-Confidence 回测（论文最终版）')
    parser.add_argument('--base',         default='output',
                        help='模型输出目录根路径')
    parser.add_argument('--runs',         type=int,   default=3,
                        help='平均的 run 数')
    parser.add_argument('--vol_quantile', type=float, default=0.75,
                        help='置信度分位阈值（建议 0.70~0.80）')
    parser.add_argument('--cost_bps',     type=float, default=0.002,
                        help='每次换手扣除的标准差比例（0.002=0.2%%σ/trade）')
    args = parser.parse_args()

    # ── 跑回测 ────────────────────────────────────────────────────────
    results = run_all(args)

    # ── 保存 JSON ─────────────────────────────────────────────────────
    save_json(results)

    # ── 图1：Proposed vs SOTA Baselines ───────────────────────────────
    plot_group(
        results,
        model_group=BASELINES_GROUP,
        title='Out-of-Sample PnL: Proposed vs SOTA Baselines'
              '\n(Vol-Confidence Strategy, q=0.75)',
        out_path='backtest_baselines.png',
        args=args,
    )

    # ── 图2：消融实验 ─────────────────────────────────────────────────
    plot_group(
        results,
        model_group=ABLATION_GROUP,
        title='Out-of-Sample PnL: Ablation Study'
              '\n(M0→M3 Incremental Architecture)',
        out_path='backtest_ablations.png',
        args=args,
    )

    print('\n[完成] 请查看:')
    print('  backtest_baselines.png   ← 论文图1：Proposed vs Baselines')
    print('  backtest_ablations.png   ← 论文图2：消融实验')
    print('  backtest_metrics.json    ← 结构化指标')
