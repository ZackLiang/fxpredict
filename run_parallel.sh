#!/bin/bash
# run_parallel.sh —— 单卡并行版（H20 140GB 显存，9 个任务同时跑）
#
# 用法：
#   bash run_parallel.sh              # 默认 30 epoch，3 runs
#   EPOCHS=100 RUNS=5 bash run_parallel.sh
#
# 原理：所有任务共享同一块 GPU（cuda:0），PyTorch 会自动分时占用。
# 140GB 显存 / 每个任务峰值约 2~4GB = 并行 9 个绰绰有余。
# 所有子进程后台运行，最后 wait 统一等待全部完成。

set -e
cd "$(dirname "$0")"

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-30}"
RUNS="${RUNS:-3}"
DATA="./data/G31_RawPrice.txt"
ADJ="./data/sensor_graph/adj_mx.pkl"
NODES=31

COMMON="--data $DATA --num_nodes $NODES --epochs $EPOCHS --runs $RUNS \
        --device $DEVICE --horizon 1 --seq_in_len 168 --normalize 2 --batch_size 128"

mkdir -p ./model ./output ./logs

echo "============================================================"
echo "  并行训练启动 (单卡 H20，140GB 显存)"
echo "  EPOCHS=$EPOCHS  RUNS=$RUNS  DEVICE=$DEVICE"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── 第一步：生成格兰杰图（必须先跑完，M2/M3 依赖它）──────────────
echo ">>> [Pre] 生成格兰杰因果图 adj_mx.pkl ..."
python3 gen_corr_matrix.py 2>&1 | tee logs/log_granger.txt
if [ $? -ne 0 ]; then
    echo "[WARN] 格兰杰图生成失败，M2/M3 将 fallback 到 predefined_A=None"
fi
echo ">>> 格兰杰图完毕，开始并行训练..."
echo ""

# ── 第二步：9 个训练任务全部后台并发 ────────────────────────────

# --- 消融实验 M0 ---
echo ">>> 启动 M0 ..."
python3 train_single_step.py $COMMON \
  --use_mamba 0 --dual_graph 0 \
  --use_router 0 --use_cross_attn 0 \
  --use_dirloss 1 --use_diffic 1 --phys_weight 0 \
  --save ./model/model_M0.pt \
  > logs/log_Ablation_M0.txt 2>&1 &
PID_M0=$!

# --- 消融实验 M1 ---
echo ">>> 启动 M1 ..."
python3 train_single_step.py $COMMON \
  --use_mamba 1 --dual_graph 0 \
  --use_router 0 --use_cross_attn 0 \
  --use_dirloss 1 --use_diffic 1 --phys_weight 0 \
  --save ./model/model_M1.pt \
  > logs/log_Ablation_M1.txt 2>&1 &
PID_M1=$!

# --- 消融实验 M2 ---
echo ">>> 启动 M2 ..."
python3 train_single_step.py $COMMON \
  --use_mamba 1 --dual_graph 1 --adj_data $ADJ \
  --use_router 1 --use_cross_attn 1 \
  --use_dirloss 1 --use_diffic 1 --phys_weight 0 \
  --save ./model/model_M2.pt \
  > logs/log_Ablation_M2.txt 2>&1 &
PID_M2=$!

# --- 消融实验 M3（完全体）---
echo ">>> 启动 M3 ..."
python3 train_single_step.py $COMMON \
  --use_mamba 1 --dual_graph 1 --adj_data $ADJ \
  --use_router 1 --use_cross_attn 1 \
  --use_dirloss 1 --use_diffic 1 --dir_weight 0.15 --diffic_weight 0.08 --phys_weight 0.1 \
  --save ./model/model_M3_Proposed.pt \
  > logs/log_Ablation_M3.txt 2>&1 &
PID_M3=$!

# --- Baseline: LSTM ---
echo ">>> 启动 Baseline LSTM ..."
python3 train_baselines.py --model lstm $COMMON \
  --save ./model/model_lstm.pt \
  > logs/log_Baseline_lstm.txt 2>&1 &
PID_LSTM=$!

# --- Baseline: TCN ---
echo ">>> 启动 Baseline TCN ..."
python3 train_baselines.py --model tcn $COMMON \
  --save ./model/model_tcn.pt \
  > logs/log_Baseline_tcn.txt 2>&1 &
PID_TCN=$!

# --- Baseline: AGCRN ---
echo ">>> 启动 Baseline AGCRN ..."
python3 train_baselines.py --model agcrn $COMMON \
  --save ./model/model_agcrn.pt \
  > logs/log_Baseline_agcrn.txt 2>&1 &
PID_AGCRN=$!

# --- Baseline: PatchTST ---
echo ">>> 启动 Baseline PatchTST ..."
python3 train_baselines.py --model patchtst $COMMON \
  --save ./model/model_patchtst.pt \
  > logs/log_Baseline_patchtst.txt 2>&1 &
PID_PATCHTST=$!

# --- Baseline: iTransformer ---
echo ">>> 启动 Baseline iTransformer ..."
python3 train_baselines.py --model itransformer $COMMON \
  --save ./model/model_itransformer.pt \
  > logs/log_Baseline_itransformer.txt 2>&1 &
PID_ITRANS=$!

echo ""
echo "============================================================"
echo "  9 个任务已全部后台启动，PID 列表："
echo "  M0=$PID_M0  M1=$PID_M1  M2=$PID_M2  M3=$PID_M3"
echo "  LSTM=$PID_LSTM  TCN=$PID_TCN  AGCRN=$PID_AGCRN"
echo "  PatchTST=$PID_PATCHTST  iTransformer=$PID_ITRANS"
echo "  实时监控（另开终端执行）:"
echo "    watch -n 10 nvidia-smi"
echo "    tail -f logs/log_Ablation_M3.txt"
echo "============================================================"
echo ""

# ── 第三步：等待所有任务完成，逐一汇报退出状态 ────────────────────
declare -A PIDS=(
    [M0]=$PID_M0 [M1]=$PID_M1 [M2]=$PID_M2 [M3]=$PID_M3
    [LSTM]=$PID_LSTM [TCN]=$PID_TCN [AGCRN]=$PID_AGCRN
    [PatchTST]=$PID_PATCHTST [iTransformer]=$PID_ITRANS
)

ALL_OK=1
for NAME in "${!PIDS[@]}"; do
    PID=${PIDS[$NAME]}
    wait $PID
    CODE=$?
    if [ $CODE -eq 0 ]; then
        echo "  [✓] $NAME 完成"
    else
        echo "  [✗] $NAME 失败 (exit=$CODE)，检查 logs/log_*${NAME}*.txt"
        ALL_OK=0
    fi
done

echo ""
echo "============================================================"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
if [ $ALL_OK -eq 1 ]; then
    echo "  所有实验成功完成！"
else
    echo "  部分实验失败，请检查对应 log 文件"
fi
echo "============================================================"
echo ""

# ── 第四步：自动提取所有模型的最终指标 ──────────────────────────
echo "============================================================"
echo "  最终指标汇总（从 logs/ 提取 Summary 块）"
echo "============================================================"
printf "%-16s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n" \
    "Model" "MAE" "RMSE" "DA" "tDA" "IC" "ICIR"
printf "%-16s-+-%-8s-+-%-8s-+-%-8s-+-%-8s-+-%-8s-+-%-8s\n" \
    "----------------" "--------" "--------" "--------" "--------" "--------" "--------"

for LOG in logs/log_Ablation_M0.txt logs/log_Ablation_M1.txt \
           logs/log_Ablation_M2.txt logs/log_Ablation_M3.txt \
           logs/log_Baseline_lstm.txt logs/log_Baseline_tcn.txt \
           logs/log_Baseline_agcrn.txt logs/log_Baseline_patchtst.txt \
           logs/log_Baseline_itransformer.txt; do
    [ -f "$LOG" ] || continue
    NAME=$(basename $LOG .txt | sed 's/log_Ablation_//' | sed 's/log_Baseline_//')
    # 从 Summary 块提取均值（格式：MAE | 0.0123 | 0.0004）
    MAE=$(grep -i  "^MAE"  "$LOG" | tail -1 | awk -F'|' '{gsub(/ /,"",$2); print $2}')
    RMSE=$(grep -i "^RMSE" "$LOG" | tail -1 | awk -F'|' '{gsub(/ /,"",$2); print $2}')
    DA=$(grep -i   "^DA"   "$LOG" | tail -1 | awk -F'|' '{gsub(/ /,"",$2); print $2}')
    TDA=$(grep -i  "^TDA"  "$LOG" | tail -1 | awk -F'|' '{gsub(/ /,"",$2); print $2}')
    IC=$(grep -i   "^IC"   "$LOG" | tail -1 | awk -F'|' '{gsub(/ /,"",$2); print $2}')
    ICIR=$(grep -i "^ICIR" "$LOG" | tail -1 | awk -F'|' '{gsub(/ /,"",$2); print $2}')
    printf "%-16s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n" \
        "$NAME" "${MAE:--}" "${RMSE:--}" "${DA:--}" "${TDA:--}" "${IC:--}" "${ICIR:--}"
done

echo ""
echo "  详细日志位于 logs/ 目录"
echo "  下一步：python3 backtest.py && python3 plot_results.py"
