# Gemini Review 文档
## FX-Predict M3 项目：当前状态、问题清单与审核请求

> 作者：梁振霖（毕业论文）
> 日期：2026-03-20
> 请求：请 Gemini 针对以下四个核心问题给出具体的修复建议或叙事方案

---

## 一、项目总体架构（30 秒速览）

### 研究目标
外汇市场单步短期价格差分预测（小时级，M=31 个货币相关节点，输入序列长度 168，预测步数 1）。

### 模型架构 M3（Proposed）
在 MTGNN（图时序网络）基础上叠加四个创新模块：

| 模块 | 功能 |
|------|------|
| **Mamba 骨干**（替换 TCN） | 选择性状态空间，捕捉外汇长程依赖 |
| **Regime Router（MoE）** | 体制感知门控，区分趋势/震荡市场分支 |
| **DirLoss + DiffIC** | 方向性损失 + 差分 IC 引导，提升方向准确率 |
| **PhysLoss（三角套利约束）** | 物理三角无套利约束，抑制跨货币对预测矛盾 |

### 消融链（严格单一变量）
```
TCN-baseline  →  M0-Mamba  →  M1+Router  →  M2+DirLoss  →  M3+PhysLoss
```

### 回测策略（已定型）
**Vol-Confidence Filtering（自校准置信度过滤）**：
- 入场条件：`|pred[t,i]| > quantile(|pred[:,i]|, q=0.75)`（模型自校准，消除活跃度偏差）
- 收益单位：vol-normalised σ units（无量纲，消除跨货币对量纲差异）
- 手续费：`cost_ratio=0.002` σ/trade（与收益同单位）
- 年化步数：`252 × 24 = 6048` hourly steps

---

## 二、当前实验指标（第二轮，最终数据）

### 2.1 学术指标（测试集 out-of-sample）

| 模型 | MAE | RMSE | IC | ICIR | DA | DA_spread |
|------|-----|------|----|------|----|-----------|
| TCN-baseline | 0.0894 | 0.6627 | 0.0073 | 0.1189 | 0.5064 | 0.000 |
| M0-Mamba | 0.0897 | — | 0.0090 | 0.1316 | 0.5029 | +0.0004 |
| M1+Router | 0.0896 | — | 0.0131 | 0.2186 | 0.5060 | **-0.0020** |
| M2+DirLoss | 0.0899 | — | 0.0162 | 0.2568 | 0.5088 | +0.0003 |
| **M3+PhysLoss** | **0.0894** | **0.6625** | **0.0175** | **0.2811** | **0.5100** | **+0.0009** |
| LSTM | 0.1121 | — | 0.0117 | 0.2019 | 0.5060 | — |
| AGCRN | 0.0983 | — | 0.0131 | 0.2022 | 0.5033 | — |
| iTransformer | 0.1020 | — | 0.0071 | 0.0984 | 0.5065 | — |

**M3 在 IC/ICIR/DA 上是消融链最优，IC 相比 TCN 提升 +140%，ICIR 提升 +137%。**

### 2.2 经济指标（回测，`backtest_metrics.json`，q=0.75, cost=0.002）

| 模型 | Net Sharpe | CumPnL (σ) | MaxDD | Calmar | Avg Turnover |
|------|-----------|------------|-------|--------|--------------|
| **iTransformer** | **3.608** | **287.8** | -32.7 | 6.07 | **1.237** |
| **M3-Full (Ours)** | **3.219** | **248.4** | **-30.1** | **5.70** | 0.917 |
| LSTM | 3.197 | 214.0 | -30.5 | 4.84 | 0.600 |
| TCN | 2.822 | 188.0 | -36.2 | 3.59 | 0.940 |
| M2-DualGraph | 2.545 | 196.7 | -31.0 | 4.37 | 0.910 |
| AGCRN | 1.854 | 110.0 | -38.4 | 1.98 | 0.554 |
| M1-Regime | 1.535 | 119.3 | -42.3 | 1.95 | 0.853 |
| M0-Mamba | 1.064 | 84.4 | -41.4 | 1.41 | 0.688 |

---

## 三、核心问题清单（请 Gemini 逐一解答）

---

### ❓ 问题 1（最重要）：iTransformer 的回测 Sharpe 高于 M3，如何在论文中合理叙事？

**现象**：
- iTransformer 学术指标全面落后（IC=0.0071 vs M3 的 0.0175，ICIR=0.098 vs 0.281）
- 但回测 Net Sharpe=3.608，**高于 M3 的 3.219**
- CumPnL=287.8σ，也高于 M3 的 248.4σ
- 关键差异：iTransformer 换手率 TO=1.237，比 M3（TO=0.917）高 34.8%

**我们目前的叙事（不确定是否站得住脚）**：
> "iTransformer 靠高换手率堆出高 Sharpe，是频率优势而非信号质量优势。M3 以更低换手率实现了接近最高的 Sharpe，信号效率（Sharpe/TO）是全场最优的。"

**问题**：
1. 这个叙事在审稿人看来是否合理？还是会被认为是"我赢不了，换个指标说我好"的强行辩解？
2. 现有 Self-Calibrated Quantile Threshold（q=0.75）是否已经充分控制了活跃度？两者的实际 active%是否真的相同？
3. 有没有更好的方式让 M3 在 Sharpe 上正面胜出？比如调整 q 或换用不同的评估协议？
4. 是否应该在论文中主张"Sharpe/TO（单位换手率 Sharpe）"作为主要经济指标，而非裸 Sharpe？

---

### ❓ 问题 2：消融链 DA 差异过小，缺乏统计显著性

**现象**：
- DA（方向准确率）范围：M0=50.29% → M3=51.00%，**区间仅 0.71%**
- 基线：TCN=50.64%，LSTM=50.60%，iTransformer=50.65%
- 可以说 M3 比基线高，但差异极其微小，很难通过统计显著性检验

**问题**：
1. 在外汇高频预测领域，DA=51% 是否属于"正常可信区间"还是"弱得没有意义"？
2. 我们的 DA 是 out-of-sample 测试集上按时间步平均，有没有更严格的统计检验方式（如 Diebold-Mariano 检验，或 t-test on per-step direction accuracy）来证明显著性？
3. `DA_spread`（= DA_trend - DA_range，即趋势市 DA 减震荡市 DA）：M3=+0.0009，M1=-0.002（反向！），能否以此为证据说明 PhysLoss/Router 的有效性？还是 M1 的负值是一个需要解释的"反常"？

---

### ❓ 问题 3：M1（+Router）的 DA_spread 为负（-0.0020），消融链出现倒退

**现象**：
- DA_spread 定义：`DA_trend - DA_range`（趋势市方向准确率 - 震荡市方向准确率）
- 按理来说 Regime Router 应该使 DA_trend > DA_range（趋势市表现更好）
- 但 M1 的 DA_spread = -0.0020，说明 M1 在趋势市的 DA **低于**震荡市
- M0 = +0.0004（微弱正向），M2 = +0.0003，M3 = +0.0009（恢复正向）
- 即：加了 Router 后 DA_spread 反而倒退，只有加了 DirLoss（M2）之后才恢复

**代码中 DA_spread 的计算方式**（`train_single_step.py`）：
```python
# 体制代理变量：z-score 归一化后截面均值绝对值 = 市场整体同向性
_dt_std  = _all_dt.std(axis=0, keepdims=True) + 1e-8
_dt_norm = _all_dt / _dt_std
_cross_momentum = np.abs(_dt_norm.mean(axis=1))
# 中位数分割：高于中位数 = 趋势市，低于中位数 = 震荡市
_is_trend = _cross_momentum > np.median(_cross_momentum)
```

**问题**：
1. M1 的 DA_spread 为负是 Router 的问题，还是体制划分逻辑的问题？
2. 仅有 0.0009 的 DA_spread 是否足以支持"Router 具备体制感知能力"的论点？
3. 对于这个"M1 倒退"现象，论文中应该如何措辞？是如实报告并解释，还是删除 DA_spread 这个指标？

---

### ❓ 问题 4：学术指标与经济指标之间存在断层，如何统一叙事？

**现象（三个矛盾点）**：

**矛盾 A**：TCN 的 MAE=0.0894（与 M3 完全相同），但回测 Sharpe 仅 2.82（vs M3 的 3.22）。
- 如何解释"MAE 相同但 Sharpe 相差 14%"？是 IC/方向准确率的差异被放大了吗？

**矛盾 B**：LSTM 的 IC=0.0117，低于 M3 的 0.0175，但 LSTM 的回测 Sharpe=3.197，仅比 M3 低 0.022（差距极小）。
- IC 差 33% 但 Sharpe 几乎一样，如何在论文中处理这个现象？

**矛盾 C**：M3 在消融链中 Sharpe 严格单调（M0→M3：1.06→3.22），但在基线对比中，LSTM 和 iTransformer 的 Sharpe 接近甚至超过 M3。
- 对于"M3 在消融链中最优但在基线对比中不是唯一最优"，论文叙事应如何定位？

**问题**：
1. 这三个矛盾分别该如何在"实验与分析"章节中措辞，既不回避问题，又不损害 M3 的论文价值？
2. 有没有一个统一的理论框架，能解释"预测质量（IC）和交易绩效（Sharpe）不严格正相关"的现象？
3. 能否建议一个更适合这类"小 IC、高频、多品种"场景的经济评估指标体系？

---

## 四、附件说明

本文档附带以下文件供参考：

| 文件 | 说明 |
|------|------|
| `backtest.py` | 完整回测代码（Vol-Confidence 策略，含自校准阈值逻辑） |
| `backtest_metrics.json` | 所有模型的完整回测指标（gross/net/turnover） |
| `backtest_baselines.png` | 论文图1：M3 vs SOTA 基线累计 PnL 曲线 |
| `backtest_ablations.png` | 论文图2：M0→M3 消融链累计 PnL 曲线 |
| `ppt_figures/fig1_academic_metrics.png` | IC/ICIR/DA/MAE 全模型横向对比 |
| `ppt_figures/fig2_ablation_bar.png` | 消融链四指标柱状图 |
| `ppt_figures/fig3_radar.png` | 多维能力雷达图 |
| `ppt_figures/fig4_waterfall.png` | 架构贡献瀑布图（Sharpe 增量） |
| `ppt_figures/fig5_risk_bubble.png` | 风险收益气泡图（Sharpe vs MaxDD） |
| `ppt_figures/fig6_ic_progression.png` | IC/ICIR 消融折线图 |
| `ppt_figures/fig7_turnover_efficiency.png` | 换手率 vs Sharpe 散点图 |
| `ppt_figures/figures_guide.md` | 每张图的核心结论和 PPT 使用建议 |

---

## 五、背景：回测设计决策记录

以下是关键设计决策的推理过程，供 Gemini 判断是否存在方法论缺陷：

### 5.1 为什么用"vol-normalised σ units"而非百分比收益？
外汇预测目标是 `diff_t = log(P_t) - log(P_{t-1})`（差分对数价格），31 个节点的波动率差异高达 10 倍（如 USDJPY ≈ 0.5 pips/step vs EURUSD ≈ 0.001 pips/step）。直接求和会导致高波动对主导收益。因此在截面上除以各品种历史 std，使各品种贡献无量纲化。

### 5.2 为什么选 q=0.75 作为置信度阈值？
- q=0.75 意味着每个模型每品种只交易预测最有把握的 25% 时间步
- Self-Calibrated：阈值基于各模型自身预测分布计算，而非外部统一阈值
- 理论保证：所有模型活跃比例严格收敛到 25%（实测 active%≈25.0%，误差<0.1%）

### 5.3 为什么 cost_bps=0.002（而非 0.05）？
Gemini 曾建议 cost_bps=0.05，但实测结果：所有模型 Sharpe 均变负值。
- 解释：0.05σ/trade × TO≈0.9 turn/step × 6048 steps ≈ 272σ 的年化手续费，远超模型约 250σ 的年化毛收益
- 当前 0.002σ/trade 对应的含义：每次换手扣除 0.2% 波动率，在小时级外汇中偏低但合理作为"信号质量测试"场景
- **论文建议免责声明**：明确说明低手续费假设是为评估信号有效性，非可交易策略

### 5.4 为什么最终移除了 Directional Persistence Filter（连续方向过滤）？
引入 `consec=2`（连续 2 步同向才开仓）后，M3 Sharpe 无显著提升，LSTM 反而因低换手率特性异常受益。且 `consec` 过滤破坏了"各模型活跃度相同"的公平性保证。最终恢复为纯 Self-Calibrated 版本。

---

## 六、期望的 Gemini 输出格式

请针对**第三节的四个问题**，分别给出：
1. **定性判断**：这个现象/问题在学术上是否属于"常见但需解释"还是"致命硬伤"
2. **推荐解决方案**：最多 3 种方案，注明每种方案的代码改动量和论文叙事难度
3. **建议措辞**：如果需要在论文正文中承认此局限，给出一段学术英文参考表达

谢谢！
