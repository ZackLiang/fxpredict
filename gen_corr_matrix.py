"""
gen_corr_matrix.py
构建"格兰杰因果"静态先验图（有向），替代原来的皮尔逊相关图。

格兰杰因果与皮尔逊相关的本质区别：
  - 皮尔逊: A ↔ B（对称，MTGNN 自适应图已经能学到）
  - 格兰杰: A → B（有向）= "A 的历史能否改善对 B 未来的预测"
  - MTGNN 的自适应图只看当前 batch 节点嵌入，天然无法捕捉跨时间有向因果
  → 两者互补，静态格兰杰图提供的信号是增量信息
"""
import pandas as pd
import numpy as np
import pickle
import os
from statsmodels.tsa.stattools import grangercausalitytests

# === 配置参数 ===
DATA_PATH   = './data/G31_RawPrice.txt'
OUTPUT_PATH = './data/sensor_graph/adj_mx.pkl'
TRAIN_RATIO = 0.6    # 只用训练集，防止数据泄露
MAXLAG      = 3      # 检验滞后 1-3 期
P_THRESH    = 0.10   # 显著性阈值：p < 0.10 才认为存在因果关系
TOP_K       = 8      # 每个节点最多保留 Top-8 条入边


def compute_granger_matrix(data: np.ndarray, maxlag: int = 3, p_thresh: float = 0.10) -> np.ndarray:
    """
    计算有向格兰杰因果矩阵。
    adj[i, j] = i → j 的因果强度（F 统计量，不显著则为 0）
    """
    n = data.shape[1]
    adj = np.zeros((n, n), dtype=np.float32)
    total = n * (n - 1)
    done = 0

    for j in range(n):          # j 是被预测的目标列
        for i in range(n):      # i 是候选"原因"列
            if i == j:
                continue
            # statsmodels 约定：第 0 列为 y（被预测），第 1 列为 x（原因）
            xy = np.column_stack([data[:, j], data[:, i]])
            try:
                res = grangercausalitytests(xy, maxlag=maxlag, verbose=False)
                # 取各滞后中最小 p 值及对应 F 统计量
                min_p  = min(res[l][0]['ssr_ftest'][1] for l in range(1, maxlag + 1))
                max_f  = max(res[l][0]['ssr_ftest'][0] for l in range(1, maxlag + 1))
                if min_p < p_thresh:
                    adj[i, j] = max_f   # i 对 j 有显著因果，权重 = F 统计量
            except Exception:
                pass

            done += 1
            if done % 50 == 0:
                print(f"  格兰杰检验进度: {done}/{total} ({100*done/total:.0f}%)", flush=True)

    return adj


def gen_granger_graph():
    print(f"读取数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, header=None)
    print(f"数据形状: {df.shape}")

    # 只使用训练集（前 60%），防止数据泄露
    train_rows = int(len(df) * TRAIN_RATIO)
    df_train = df.iloc[:train_rows]
    print(f"仅使用训练集: {train_rows}/{len(df)} 行")

    # 对数收益率（平稳化，减少"伪因果"）
    df_returns = np.log(df_train.clip(lower=1e-8)).diff().dropna()
    data = df_returns.values
    print(f"收益率数据形状: {data.shape}")
    print(f"开始格兰杰因果检验 (maxlag={MAXLAG}, p<{P_THRESH})，共 {data.shape[1]*(data.shape[1]-1)} 次检验...\n")

    adj = compute_granger_matrix(data, maxlag=MAXLAG, p_thresh=P_THRESH)

    # Top-K 稀疏化：每个目标节点只保留最强的 TOP_K 条入边
    n = adj.shape[0]
    sparse = np.zeros_like(adj)
    for j in range(n):
        col = adj[:, j]                         # 所有 i→j 的权重
        top_k_idx = np.argsort(col)[-TOP_K:]    # 最强的 TOP_K 个原因
        sparse[top_k_idx, j] = col[top_k_idx]

    print(f"\n格兰杰因果图（Top-{TOP_K}）前 5×5：")
    print(sparse[:5, :5])

    n_edges = np.sum(sparse > 0)
    print(f"非零边数: {n_edges} / {n*n}  (密度 {100*n_edges/n/n:.1f}%)")

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(sparse.astype(np.float32), f)
    print(f"\n已保存到: {OUTPUT_PATH}")


if __name__ == '__main__':
    gen_granger_graph()
