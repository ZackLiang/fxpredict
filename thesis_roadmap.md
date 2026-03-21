# 毕业论文实验全景梳理
## 基于多图融合的外汇市场方向性预测模型

---

## 当前实验设计（2025 更新）

> **核心脚本**：`run_experiments.sh`（严格分离 Baselines 主对比 与 消融实验）
> **消融链**：M0(TCN) → M1(Mamba) → M2(+MoE) → M3(+PhysLoss)

### 三级火箭故事线

| 痛点 | 解法 | 消融变体 |
|------|------|----------|
| ① 长程记忆衰减 | Mamba 替换 TCN | M0→M1 |
| ② 震荡/趋势频繁切换 | Regime-MoE 体制感知 | M1→M2 |
| ③ 黑盒违背套利常识 | Physical Consistency Loss | M2→M3 |

### 运行方式

```bash
# 完整实验（Baselines + 消融 M0~M3）
./run_experiments.sh

# Mac MPS / 自定义 epoch
DEVICE=mps EPOCHS=30 ./run_experiments.sh
```

### 输出文件

| 类型 | 路径 |
|------|------|
| Baseline 日志 | `log_Baseline_{lstm,tcn,agcrn,itransformer}.txt` |
| Proposed 日志 | `log_Proposed.txt` |
| 消融日志 | `log_Ablation_M0.txt` ~ `log_Ablation_M3.txt` |
| 模型权重 | `model/model_*.pt` |
| 预测文件 | `output/model_*/diff_pred_run*.npy`（backtest 用）|

---

## 一、论文核心故事

### 一句话概括

> 外汇市场存在**三个建模痛点**：长程依赖衰减、震荡与趋势体制切换、黑盒预测违背三角套利常识。
> 本文通过 Mamba 骨干、Regime-MoE 体制路由、Physical Consistency Loss 三个递进创新，
> 在**方向性指标（DA/IC/ICIR）**上系统优于 Baselines，并在理论套利框架下证明可转化为交易盈利。

### 叙事主轴

```
第一幕：问题提出
  外汇 = 长程宏观 + 体制切换 + 套利约束
  → 现有模型（TCN/LSTM/Transformer）长程衰减、忽视方向性

第二幕：创新设计（三级火箭）
  M1: Mamba 替换 TCN → 解决长程记忆衰减
  M2: Regime-MoE    → 体制感知，抗震荡
  M3: PhysLoss      → 物理约束，套利逻辑闭环

第三幕：实验验证
  ① 消融：M0→M1→M2→M3 逐步提升
  ② 主对比：Proposed (M3) vs LSTM/TCN/AGCRN/iTransformer
  ③ 交易策略：IC/DA 提升 → 理论盈利能力

第四幕：结论
  图结构 + 体制感知 + 物理约束 = 外汇预测正确打开方式
```

---

## 二、消融实验设计

### 消融链（论文 4.1 节）

| 变体 | 时序骨干 | 双图 | MoE | PhysLoss | 对应创新 |
|------|----------|------|-----|----------|----------|
| **M0** | TCN | 否 | 否 | 否 | 基线（原始 MTGNN） |
| **M1** | Mamba | 否 | 否 | 否 | 创新① 长程记忆 |
| **M2** | Mamba | 是 | 是 | 否 | 创新② 体制感知 |
| **M3** | Mamba | 是 | 是 | 是 | 创新③ 物理约束（完全体） |

### 预期效果

- **M0→M1**：MAE/RMSE 下降（长程预测更准）
- **M1→M2**：DA_SPREAD 改善、抗震荡增强
- **M2→M3**：CCC 提升、夏普转正（金融逻辑闭环）

### 关键参数

```bash
# M0 (TCN 基线)
--use_mamba 0 --dual_graph 0 --use_router 0 --phys_weight 0

# M1 (+ Mamba)
--use_mamba 1 --dual_graph 0 --use_router 0 --phys_weight 0

# M2 (+ MoE)
--use_mamba 1 --dual_graph 1 --use_router 1 --use_cross_attn 1 --phys_weight 0

# M3 (完全体)
--use_mamba 1 --dual_graph 1 --use_router 1 --phys_weight 0.1
```

---

## 三、主对比实验（Baselines vs Proposed）

### 对手模型

| 模型 | 类型 | 来源 |
|------|------|------|
| LSTM | 纯序列 | train_baselines.py |
| TCN | 时序卷积 | train_baselines.py |
| AGCRN | 自适应图 | train_baselines.py |
| iTransformer | 变量注意力 | train_baselines.py |
| **Proposed** | M3 完全体 | train_single_step.py |

### 验证目标

- **MAE**：Proposed ≤ Baselines（图结构精度优势）
- **IC/DA**：Proposed > Baselines（方向性训练目标对齐）
- **策略盈利**：Proposed 在 CS-Momentum 等策略下 Sharpe > 0

---

## 四、代码文件说明

| 文件 | 用途 |
|------|------|
| `net.py` | gtnet（--use_mamba 切换 TCN/Mamba）、Baseline 模型 |
| `train_single_step.py` | 消融训练（M0~M3 全参数） |
| `train_baselines.py` | LSTM/TCN/AGCRN/iTransformer 训练 |
| `run_experiments.sh` | **主脚本**：Baselines + 消融 全自动 |
| `backtest.py` | 回测引擎 |
| `gen_corr_matrix.py` | 格兰杰图 adj_mx.pkl 生成 |
| `build_dataset.py` | G31_RawPrice.txt 从 CSV 构建 |

---

## 五、待做事项

1. **跑满实验**：`./run_experiments.sh`（建议 `RUNS=3`）
2. **论文表格**：从 `log_Ablation_M*.txt`、`log_Baseline_*.txt` 提取指标
3. **回测验证**：`backtest.py` 使用 `output/model_*/` 预测文件
4. **绘图**：`plot_results.py`、`plot_bars.py` 需按新 M0~M3 命名适配
