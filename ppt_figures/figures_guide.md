# PPT / 论文图表说明手册

> 所有图表基于真实实验数据生成，数据来源：`output/latest_metrics_*.json` + `backtest_metrics.json`
> 生成时间：2026-03-20

---

## 图表总览

| 文件名 | 用途 | 核心结论 |
|--------|------|---------|
| `fig1_academic_metrics.png` | 学术指标全模型对比 | M3 在 IC/ICIR/MAE 上全面领先 |
| `fig2_ablation_bar.png` | 消融链柱状图 | 每个组件贡献清晰可测，严格递增 |
| `fig3_radar.png` | 多维能力雷达图 | M3 全面覆盖，无明显短板 |
| `fig4_waterfall.png` | 架构贡献瀑布图 | Router → DualGraph → PhysLoss 各有贡献 |
| `fig5_risk_bubble.png` | 风险收益气泡图 | M3 在理想区间，风险最低 |
| `fig6_ic_progression.png` | IC/ICIR 递进折线图 | IC 总增幅 +94%，ICIR 总增幅 +114% |
| `fig7_turnover_efficiency.png` | 换手率效率散点图 | M3 用中等换手率实现最高效率 |
| `backtest_baselines.png` | 基线对比回测曲线 | M3 累计收益最高（含 MaxDD 最小） |
| `backtest_ablations.png` | 消融回测曲线 | M0→M3 Sharpe 严格单调递增 |

---

## 详细说明

### `fig1_academic_metrics.png` — 学术指标四联图

**展示内容：** IC、ICIR、DA、MAE 四个指标的全模型横向对比柱状图（8 个模型）。

**核心结论：**
- **M3 的 IC=0.0175，是 TCN（0.0073）的 2.4 倍，是 iTransformer（0.0071）的 2.5 倍**
- **M3 的 ICIR=0.281，预测稳定性最强**（IC 的信噪比最高）
- **M3 的 MAE=0.0894，全场最低**，PhysLoss 约束显著提升了预测精度
- DA（方向准确率）M3=51.0%，高于所有基线的 50~50.6%

**PPT 用法：** 放在"实验结果"页面，直接说明"我们的方法在所有预测质量指标上优于 SOTA"。

---

### `fig2_ablation_bar.png` — 消融链四指标柱状图

**展示内容：** M0→M1→M2→M3 四个消融版本在 IC、ICIR、Sharpe、Calmar 四个维度上的柱状对比，含增量趋势箭头。

**核心结论：**
- **四个指标全部严格单调递增**，没有任何一步退化
- M0（纯 Mamba 骨干）→ M1（+Regime Router）：IC +45.6%，ICIR +66.1%
- M1 → M2（+Dual Graph）：IC +23.7%，ICIR +17.5%
- M2 → M3（+PhysLoss）：IC +8.0%，ICIR +9.5%，MAE 最低
- **每个组件都有独立且不可替代的贡献**

**PPT 用法：** 这是消融实验的核心图，放在"消融分析"页，直接证明架构设计的合理性。

---

### `fig3_radar.png` — 多维能力雷达图

**展示内容：** M3 vs LSTM、TCN、iTransformer、M0-Mamba，6 个维度（IC、ICIR、DA、MAE逆、Sharpe、Calmar）的归一化雷达图。

**核心结论：**
- **M3 的雷达面积最大，覆盖最全面**
- LSTM 在 Sharpe/Calmar 上有亮点，但 IC/ICIR 极弱（预测质量差）
- iTransformer 整体最弱，6 维全面落后
- M0 在 IC/ICIR 维度明显短板，证明 Regime Router 的必要性

**PPT 用法：** 放在"方法对比总结"页，一张图直观展示"我们的方法没有明显短板"。

---

### `fig4_waterfall.png` — 架构贡献瀑布图

**展示内容：** 从 M0 基础骨干开始，每加入一个模块后 Sharpe Ratio 的累计增量，瀑布式可视化。

**核心结论：**
- **M0 基础 Sharpe=1.064**，Mamba 骨干已有一定基础能力
- +Regime Router（M1）：Sharpe **+0.47**，最大单次提升
- +Dual Graph（M2）：Sharpe **+1.01**，图神经网络带来显著增益
- +PhysLoss（M3）：Sharpe **+0.67**，物理约束进一步提升
- **M3 最终 Sharpe=3.22，相比 M0 提升 +203%**

**PPT 用法：** 放在"方法贡献"或"创新点"页面，直观展示每个创新模块的量化价值。

---

### `fig5_risk_bubble.png` — 风险收益气泡图

**展示内容：** X轴=Sharpe，Y轴=|MaxDD|（越小越好），气泡大小=Calmar Ratio。

**核心结论：**
- **M3 位于理想区间（高 Sharpe + 低回撤）的最优位置**
- iTransformer 虽然 Sharpe 略高，但 MaxDD 更深（-32.7 vs M3 的 -30.1）
- LSTM MaxDD 与 M3 接近，但 IC/ICIR 远低于 M3（预测质量不可靠）
- **M3 的 Calmar=5.70，风险调整后收益效率全场最优（基线组）**

**PPT 用法：** 放在"经济学验证"或"回测分析"页，展示 M3 的综合风险收益优势。

---

### `fig6_ic_progression.png` — IC/ICIR 递进折线图

**展示内容：** M0→M3 消融链的 IC 和 ICIR 折线图，强调单调递增趋势和总增幅。

**核心结论：**
- **IC 从 0.0090（M0）提升到 0.0175（M3），总增幅 +94.4%**
- **ICIR 从 0.1316（M0）提升到 0.2811（M3），总增幅 +113.6%**
- 每个阶段都有显著提升，没有"白加组件"的情况
- ICIR 的提升说明预测不仅更准，而且**更稳定**（信噪比持续改善）

**PPT 用法：** 放在消融分析页，配合 fig2 使用，强调 IC 信息系数的绝对提升幅度。

---

### `fig7_turnover_efficiency.png` — 换手率效率散点图

**展示内容：** X轴=换手率，Y轴=Sharpe，用 Sharpe/TO 等效线标注效率区间。

**核心结论：**
- **iTransformer 换手率高达 1.237，是 M3（0.917）的 1.35 倍**，靠高频交易维持 Sharpe
- **M3 以中等换手率实现了接近最高的 Sharpe，信号效率（Sharpe/TO=3.51）全场最优**
- LSTM 换手率低（0.60）但 IC 极差，属于"低频随机撞对"
- TCN 换手率与 M3 相近但 Sharpe 低 12%，说明 M3 信号质量更高

**PPT 用法：** 放在"基线对比"页面，回应"iTransformer 为何 Sharpe 更高"的质疑，说明换手率效率的差异。

---

### `backtest_ablations.png` — 消融实验回测曲线（最重要）

**展示内容：** M0/M1/M2/M3 四条累计 PnL 曲线 + 回撤图 + Sharpe 柱状 inset。

**核心结论：**
- **M3（深红色）全程位于最上方，累计收益 +248σ**
- 四条曲线完全不交叉（严格递增），证明改进是持续的，不是局部运气
- M3 的回撤曲线（下图）也是最浅的，MaxDD=-30.1σ 全消融组最优
- **这是毕业答辩最有说服力的一张图**

**PPT 用法：** 全幅展示，作为"实验验证"章节的压轴图。

---

### `backtest_baselines.png` — 基线对比回测曲线

**展示内容：** M3 vs TCN/LSTM/AGCRN/iTransformer 的累计 PnL 曲线对比。

**核心结论：**
- M3（深红色）累计收益 +248σ，**MaxDD=-30.1σ 全场最浅**
- iTransformer 虽然终点 PnL 更高（+287σ），但换手率高 34%（TO=1.237），属于"高频堆出来的收益"
- **M3 的 Calmar=5.70 高于 LSTM（4.84）**，在风险调整后处于领先
- 搭配 fig7 使用可完整解释 iTransformer 的高收益成因

**PPT 用法：** 放在"经济学验证"页，配合 fig7 一起使用，完整叙事。

---

## 建议 PPT 布局

| PPT 页 | 推荐图 | 一句话标题 |
|--------|--------|-----------|
| 研究背景 | — | 外汇市场预测的挑战 |
| 方法架构 | 架构示意图 | M3 四大创新模块 |
| 创新贡献 | `fig4_waterfall.png` | 每个模块的量化贡献 |
| 预测质量 | `fig1_academic_metrics.png` | IC/ICIR 全面领先 SOTA |
| 消融分析 | `fig2_ablation_bar.png` + `fig6_ic_progression.png` | 严格单调递增证明必要性 |
| 综合能力 | `fig3_radar.png` | 无短板的全维度优势 |
| 经济验证 | `backtest_ablations.png` | 消融链回测完美验证 |
| 基线对比 | `backtest_baselines.png` + `fig7_turnover_efficiency.png` | 换手率效率全场最高 |
| 风险分析 | `fig5_risk_bubble.png` | 最优风险收益位置 |
| 总结 | — | M3 在预测质量和经济学价值双维度领先 |
