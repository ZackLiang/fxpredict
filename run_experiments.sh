#!/bin/bash
# 毕业论文终极实验脚本：严格分离 Baselines 与 消融实验
#
# 第一部分：主对比实验（打别人）—— LSTM/TCN/AGCRN/iTransformer vs Proposed (M3)
# 第二部分：消融实验（拆自己）—— M0(TCN) → M1(Mamba) → M2(+MoE) → M3(+PhysLoss)
#
# 三级火箭故事线：
#   痛点1 长程记忆衰减 → M1: Mamba 替换 TCN
#   痛点2 震荡/趋势切换 → M2: Regime-MoE 体制感知
#   痛点3 黑盒违背套利 → M3: Physical Consistency Loss

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-30}"
RUNS=3
DATA="./data/G31_RawPrice.txt"
ADJ="./data/sensor_graph/adj_mx.pkl"
NODES=31

COMMON="--data $DATA --num_nodes $NODES --epochs $EPOCHS --runs $RUNS --device $DEVICE --horizon 1 --seq_in_len 168 --normalize 2"

mkdir -p ./model ./output

echo "====================================================="
echo " 第一部分：主对比实验 (Baselines vs Proposed M3)"
echo "====================================================="

for MODEL in lstm tcn agcrn itransformer; do
  echo "▶ Baseline: $MODEL ..."
  python train_baselines.py --model $MODEL $COMMON \
    --save ./model/model_${MODEL}.pt \
    2>&1 | tee log_Baseline_${MODEL}.txt
  echo "  -> $MODEL 完毕"
done

echo "▶ Proposed (M3 完全体：Mamba+MoE+PhysLoss)..."
python train_single_step.py $COMMON \
  --use_mamba 1 --dual_graph 1 --adj_data $ADJ \
  --use_router 1 --use_cross_attn 1 --use_dirloss 1 --use_diffic 1 \
  --dir_weight 0.15 --diffic_weight 0.08 --phys_weight 0.1 \
  --save ./model/model_Proposed.pt \
  2>&1 | tee log_Proposed.txt
echo "  -> Proposed 完毕"

echo ""
echo "====================================================="
echo " 第二部分：消融实验 (M0 → M1 → M2 → M3)"
echo "====================================================="

echo "▶ [1/4] M0 (Base Backbone：原始 TCN，单图，无 MoE，无 PhysLoss)..."
python train_single_step.py $COMMON \
  --use_mamba 0 --dual_graph 0 \
  --use_router 0 --use_cross_attn 0 --use_dirloss 1 --use_diffic 1 --phys_weight 0 \
  --save ./model/model_M0.pt \
  2>&1 | tee log_Ablation_M0.txt
echo "  -> M0 完毕"

echo "▶ [2/4] M1 (+ Mamba：TCN→Mamba，证明长程记忆改善)..."
python train_single_step.py $COMMON \
  --use_mamba 1 --dual_graph 0 \
  --use_router 0 --use_cross_attn 0 --use_dirloss 1 --use_diffic 1 --phys_weight 0 \
  --save ./model/model_M1.pt \
  2>&1 | tee log_Ablation_M1.txt
echo "  -> M1 完毕"

echo "▶ [3/4] M2 (+ MoE：Mamba+Regime-MoE，证明体制感知)..."
python train_single_step.py $COMMON \
  --use_mamba 1 --dual_graph 1 --adj_data $ADJ \
  --use_router 1 --use_cross_attn 1 --use_dirloss 1 --use_diffic 1 --phys_weight 0 \
  --save ./model/model_M2.pt \
  2>&1 | tee log_Ablation_M2.txt
echo "  -> M2 完毕"

echo "▶ [4/4] M3 (+ PhysLoss：完全体，证明物理约束闭环)..."
python train_single_step.py $COMMON \
  --use_mamba 1 --dual_graph 1 --adj_data $ADJ \
  --use_router 1 --use_cross_attn 1 --use_dirloss 1 --use_diffic 1 \
  --dir_weight 0.15 --diffic_weight 0.08 --phys_weight 0.1 \
  --save ./model/model_M3_Proposed.pt \
  2>&1 | tee log_Ablation_M3.txt
echo "  -> M3 完毕"

echo ""
echo "====================================================="
echo " 所有实验完成"
echo "====================================================="
echo "主对比表：log_Baseline_*.txt + log_Proposed.txt"
echo "消融表：  log_Ablation_M0.txt ~ M3.txt"
echo "  M0(TCN) → M1(Mamba) → M2(+MoE) → M3(+PhysLoss)"
