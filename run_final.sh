#!/bin/bash
# run_final.sh —— 服务器正式实验（严格单一变量消融 + Baselines）
#
# 【消融矩阵设计原则】严格单一变量控制，每步只改一个创新点：
#   TCN-baseline : 纯 TCN，无任何创新  （证明原始基线水平）
#   M0-clean     : 仅换 Mamba 骨干       （证明 Mamba 本身 vs TCN）
#   M1-clean     : Mamba + 双图 + MoE    （证明架构升级有效）
#   M2-clean     : M1 + DirLoss/DiffIC   （证明方向性 Loss 的增益）
#   M3 (Proposed): M2 + PhysLoss=0.025   （完全体，物理约束收尾）
#
# 【并行执行】消融组 M0~M3 同时并行，193GB RAM + 139GB VRAM 硬件完全支撑
# 【batch_size=128】统一控制变量，保证所有消融模型在同一梯度方差基准下可比
# 【Baselines】同时并行跑（无方向性Loss），全组统一 batch=128 保证学术严谨
# 【安全锁】OMP/MKL 线程限制 8，防止 21核 CPU 被多进程抢占崩溃

# ==========================================
# 并行安全锁：限制单进程 CPU 线程数
# 防止 8 个进程同时抢占 21 核 CPU 导致系统崩溃
# ==========================================
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

cd "$(dirname "$0")"
PY="${PY:-conda run -n fxpredict env PYTHONUNBUFFERED=1 python3 -u}"
DATA="./data/G31_RawPrice.txt"
ADJ="./data/sensor_graph/adj_mx.pkl"
DEV="${DEVICE:-cuda:0}"

# 统一超参数：消融组与 Baselines 共用，保证实验控制变量绝对对齐
# batch_size=128 保证所有模型在相同梯度更新频率下进行公平比较
COMMON="--data $DATA --num_nodes 31 --epochs 50 --runs 3 \
        --device $DEV --batch_size 128 \
        --horizon 1 --seq_in_len 168 --normalize 2"

mkdir -p ./logs ./model ./output

# ══════════════════════════════════════════════════════════════════
# 自动记账函数：从指定 JSON 文件读取指标，追加一行到 EXPERIMENT_LOG.md
# ══════════════════════════════════════════════════════════════════
log_result() {
    local MODEL_TAG="$1"
    local JSON="${2:-output/latest_metrics.json}"
    local LOG="EXPERIMENT_LOG.md"

    if [ ! -f "$JSON" ]; then
        echo "[WARN] $JSON 不存在，跳过记账 ($MODEL_TAG)"
        return
    fi

    $PY - <<PYEOF
import json, os

with open("$JSON") as f:
    m = json.load(f)

log_path = "$LOG"
model_tag = "$MODEL_TAG"

header = ""
if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
    header = """# FX-Predict 实验记录 (EXPERIMENT_LOG)

> 由 run_final.sh 自动追加，每次运行后更新。

| 时间 | 模型 | epochs | runs | batch | dir_w | diffic_w | phys_w | MAE | DA | IC | ICIR | CCC | DA_spread |
|------|------|--------|------|-------|-------|----------|--------|-----|----|----|------|-----|-----------|
"""

row = (
    f"| {m.get('timestamp','?')} "
    f"| **{model_tag}** "
    f"| {m.get('epochs','?')} "
    f"| {m.get('runs','?')} "
    f"| {m.get('batch_size','?')} "
    f"| {m.get('dir_weight','?')} "
    f"| {m.get('diffic_weight','?')} "
    f"| {m.get('phys_weight','?')} "
    f"| {m.get('mae','?')} "
    f"| {m.get('da') or 0.0:.4f} "
    f"| {m.get('ic') or 0.0:.4f} "
    f"| {m.get('icir') or 0.0:.4f} "
    f"| {m.get('ccc') or '-'} "
    f"| {m.get('da_spread', 0.0) or 0.0:+.4f} "
    f"|"
)

with open(log_path, "a") as f:
    if header:
        f.write(header)
    f.write(row + "\n")

print(f"[记账] {model_tag} 已追加到 {log_path}")
PYEOF
}

echo "============================================================"
echo "  并行消融实验 (严格单一变量控制)"
echo "  消融组 M0~M3 并行 + Baselines 并行（全组统一 batch=128）"
echo "  硬件: 139GB VRAM + 193GB RAM，8 进程并发安全可行"
echo "  每个模型: 50epoch × 3runs × batch128"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# 第一部分：Baselines 并行（统一 batch=128，与消融组控制变量对齐）
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> [$(date '+%H:%M:%S')] 并行启动 Baselines（lstm / agcrn / itransformer / tcn）..."

$PY train_baselines.py --model lstm $COMMON \
  --metrics_tag lstm \
  --save ./model/model_lstm.pt \
  > logs/log_lstm.txt 2>&1 &
PID_LSTM=$!

$PY train_baselines.py --model agcrn $COMMON \
  --metrics_tag agcrn \
  --save ./model/model_agcrn.pt \
  > logs/log_agcrn.txt 2>&1 &
PID_AGCRN=$!

$PY train_baselines.py --model itransformer $COMMON \
  --metrics_tag itransformer \
  --save ./model/model_itransformer.pt \
  > logs/log_itransformer.txt 2>&1 &
PID_ITRANS=$!

$PY train_baselines.py --model tcn $COMMON \
  --metrics_tag tcn \
  --save ./model/model_tcn.pt \
  > logs/log_tcn.txt 2>&1 &
PID_TCN=$!

echo "  Baselines PID: lstm=$PID_LSTM  agcrn=$PID_AGCRN  itransformer=$PID_ITRANS  tcn=$PID_TCN"

# ══════════════════════════════════════════════════════════════════
# 第二部分：消融组并行（M0~M3 同时启动，硬件充裕）
# 消融变量对照表：
#   模型         use_mamba  dual_graph  use_router  DirLoss  DiffIC  PhysLoss
#   M0-clean         1          0           0          0        0       0   ← 仅Mamba骨干
#   M1-clean         1          1           1          0        0       0   ← +双图+MoE
#   M2-clean         1          1           1          1        1       0   ← +方向Loss
#   M3 (Proposed)    1          1           1          1        1     0.025 ← +PhysLoss
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> [$(date '+%H:%M:%S')] 并行启动消融组（M0 / M1 / M2 / M3）..."

# ── M0-clean（仅 Mamba 骨干，无双图/MoE/Loss创新）─────────────
$PY train_single_step.py $COMMON \
  --use_mamba 1 --dual_graph 0 --use_router 0 --phys_weight 0 \
  --use_dirloss 0 --use_diffic 0 \
  --metrics_tag M0 \
  --save ./model/model_M0.pt \
  > logs/log_M0.txt 2>&1 &
PID_M0=$!

# ── M1-clean（Mamba + 双图 + MoE，仍无方向性Loss）────────────
$PY train_single_step.py $COMMON \
  --use_mamba 1 --dual_graph 1 --adj_data $ADJ \
  --use_router 1 --use_cross_attn 1 --phys_weight 0 \
  --use_dirloss 0 --use_diffic 0 \
  --metrics_tag M1 \
  --save ./model/model_M1.pt \
  > logs/log_M1.txt 2>&1 &
PID_M1=$!

# ── M2-clean（M1 + DirLoss + DiffIC，无PhysLoss）─────────────
$PY train_single_step.py $COMMON \
  --use_mamba 1 --dual_graph 1 --adj_data $ADJ \
  --use_router 1 --use_cross_attn 1 --phys_weight 0 \
  --use_dirloss 1 --use_diffic 1 \
  --dir_weight 0.04 --diffic_weight 0.02 \
  --metrics_tag M2 \
  --save ./model/model_M2.pt \
  > logs/log_M2.txt 2>&1 &
PID_M2=$!

# ── M3-Proposed（完全体，M2 + PhysLoss）──────────────────────
$PY train_single_step.py $COMMON \
  --use_mamba 1 --dual_graph 1 --adj_data $ADJ \
  --use_router 1 --use_cross_attn 1 \
  --use_dirloss 1 --use_diffic 1 \
  --dir_weight 0.04 --diffic_weight 0.02 --phys_weight 0.025 \
  --metrics_tag M3 \
  --save ./model/model_M3_Proposed.pt \
  > logs/log_M3.txt 2>&1 &
PID_M3=$!

echo "  消融组 PID: M0=$PID_M0  M1=$PID_M1  M2=$PID_M2  M3=$PID_M3"
echo ""
echo "  全部 8 个进程已在后台并行运行"
echo "  实时查看进度示例："
echo "    tail -f logs/log_M3.txt"
echo "    tail -f logs/log_lstm.txt"
echo "    watch -n 30 nvidia-smi"
echo "============================================================"

# ══════════════════════════════════════════════════════════════════
# 等待所有进程完成，再统一记账
# ══════════════════════════════════════════════════════════════════
echo ""
echo ">>> 等待消融组完成..."
wait $PID_M0;   echo ">>> [$(date '+%H:%M:%S')] M0-clean 完毕 (exit=$?)"
wait $PID_M1;   echo ">>> [$(date '+%H:%M:%S')] M1-clean 完毕 (exit=$?)"
wait $PID_M2;   echo ">>> [$(date '+%H:%M:%S')] M2-clean 完毕 (exit=$?)"
wait $PID_M3;   echo ">>> [$(date '+%H:%M:%S')] M3-Proposed 完毕 (exit=$?)"

echo ""
echo ">>> 等待 Baselines 完成..."
wait $PID_LSTM;   echo ">>> [$(date '+%H:%M:%S')] lstm 完毕 (exit=$?)"
wait $PID_AGCRN;  echo ">>> [$(date '+%H:%M:%S')] agcrn 完毕 (exit=$?)"
wait $PID_ITRANS; echo ">>> [$(date '+%H:%M:%S')] itransformer 完毕 (exit=$?)"
wait $PID_TCN;    echo ">>> [$(date '+%H:%M:%S')] tcn 完毕 (exit=$?)"

# ── 统一记账（所有进程完成后写入 EXPERIMENT_LOG.md）─────────────
echo ""
echo ">>> 开始集中记账..."
log_result "M0-clean"      "output/latest_metrics_M0.json"
log_result "M1-clean"      "output/latest_metrics_M1.json"
log_result "M2-clean"      "output/latest_metrics_M2.json"
log_result "M3-Proposed"   "output/latest_metrics_M3.json"
log_result "lstm"          "output/latest_metrics_lstm.json"
log_result "agcrn"         "output/latest_metrics_agcrn.json"
log_result "itransformer"  "output/latest_metrics_itransformer.json"
log_result "tcn-baseline"  "output/latest_metrics_tcn.json"

echo ""
echo "============================================================"
echo "  全部完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  消融链（预期单调递进）："
echo "    TCN-baseline → M0-clean → M1-clean → M2-clean → M3-Proposed"
echo ""
echo "  实验记录已写入 EXPERIMENT_LOG.md"
echo "============================================================"
