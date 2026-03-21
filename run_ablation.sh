#!/bin/bash
# run_ablation.sh —— 完整消融流程（M0~M3 + Baselines + 回测 + 绘图）
# 与 run_experiments.sh 一致，额外包含 gen_corr_matrix、backtest、plot_results

cd "$(dirname "$0")"

# ── 设备检测 ──────────────────────────────────────────────────
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    export DEVICE="${DEVICE:-cuda:0}"
elif python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    export DEVICE="${DEVICE:-mps}"
else
    export DEVICE="${DEVICE:-cpu}"
fi
export EPOCHS="${EPOCHS:-100}"

echo ">>> 设备: $DEVICE | Epochs: $EPOCHS"
echo ""

# ── 重新生成格兰杰图（防止 NumPy 版本不兼容）──────────────────
echo ">>> [0] 生成格兰杰因果图..."
python3 gen_corr_matrix.py
if [ $? -ne 0 ]; then
    echo "[WARN] 格兰杰图生成失败，M2/M3 将使用 predefined_A=None"
fi
echo ""

# ── 主实验（与 run_experiments.sh 相同逻辑）──────────────────
echo ">>> [1] 运行实验（M0~M3 + Baselines + Proposed）..."
./run_experiments.sh
echo ""

# ── 回测 ──────────────────────────────────────────────────────
echo ">>> [2] 金融回测..."
python3 backtest.py --runs 3 --cost_bps 1.0
echo ""

# ── 论文图表 ───────────────────────────────────────────────────
echo ">>> [3] 生成论文图表..."
python3 plot_results.py 2>/dev/null || python3 plot_results.py --demo
echo ">>> 图表已保存至 ppt_figures/"
echo ""
echo "=================================================="
echo " 全部完成"
echo "=================================================="
