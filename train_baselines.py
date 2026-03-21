# -*- coding: utf-8 -*-
"""
train_baselines.py
对比实验 Baseline 模型训练脚本。

复用 train_single_step.py 中完全相同的：
  - DataLoaderS（相同 train/val/test split + 相同 normalization）
  - evaluate()  （相同 DA/tDA/IC/ICIR 计算口径）
  - Optim       （Adam，相同超参）
  - 保存 diff_pred_run{i}.npy 供 backtest.py 直接使用

支持模型：
  --model  lstm | tcn | agcrn | patchtst | itransformer

示例：
  python3 train_baselines.py --model lstm        --save ./model/model_LSTM.pt
  python3 train_baselines.py --model tcn         --save ./model/model_TCN.pt
  python3 train_baselines.py --model agcrn       --save ./model/model_AGCRN.pt
  python3 train_baselines.py --model patchtst    --save ./model/model_PatchTST.pt
  python3 train_baselines.py --model itransformer --save ./model/model_iTransformer.pt

所有指标与 train_single_step.py 完全对齐（相同数据，相同评估函数），
保证"公平对比"。
"""
import argparse
import math
import time
import os
import torch
import torch.nn as nn
import numpy as np

# ── 复用现有模块 ─────────────────────────────────────────────────────────
from util    import DataLoaderS
from net     import (LSTMBaseline, TCNBaseline, AGCRNBaseline,
                     PatchTSTBaseline, iTransformerBaseline)


# ══════════════════════════════════════════════════════════════════════
# 1. evaluate()：与 train_single_step.py 完全相同逻辑，直接复制确保口径一致
#    唯一区别：模型接口统一，不需要处理 gtnet 的特殊 num_split 逻辑
# ══════════════════════════════════════════════════════════════════════
def evaluate(data, X, Y, LP, model, evaluateL2, evaluateL1, batch_size,
             save_dir=None, run_id=None):
    """
    评估函数（与 train_single_step.py 完全口径一致）。
    X, Y, LP = data.train/valid/test 三元素；LP = last_price。
    MAE/RMSE 在绝对价格空间；DA/IC 用差分 (output-LP)*scale。
    """
    model.eval()
    total_loss    = 0
    total_loss_l1 = 0
    n_samples     = 0
    predict = None
    test    = None

    all_diff_pred = []
    all_diff_true = []
    ic_per_batch  = []

    for X_b, Y_b, LP_b in data.get_batches(X, Y, batch_size, False, lp=LP):
        X_in = torch.unsqueeze(X_b, dim=1).transpose(2, 3)   # (B, 1, M, P)
        with torch.no_grad():
            output = model(X_in)
        output = torch.squeeze(output)
        if output.dim() == 1:
            output = output.unsqueeze(0)

        if predict is None:
            predict, test = output, Y_b
        else:
            predict = torch.cat((predict, output))
            test    = torch.cat((test, Y_b))

        scale = data.scale.expand(output.size(0), data.m)

        # MAE/RMSE：绝对价格空间
        pred_real = output * scale
        true_real = Y_b    * scale
        total_loss    += evaluateL2(pred_real, true_real).item()
        total_loss_l1 += evaluateL1(pred_real, true_real).item()

        # 差分：DA/IC
        diff_pred = (output - LP_b) * scale
        diff_true = (Y_b   - LP_b) * scale

        dp = diff_pred - diff_pred.mean(dim=1, keepdim=True)
        dt = diff_true - diff_true.mean(dim=1, keepdim=True)
        cov   = (dp * dt).mean(dim=1)
        std_p = diff_pred.std(dim=1, unbiased=False).clamp(min=1e-8)
        std_t = diff_true.std(dim=1, unbiased=False).clamp(min=1e-8)
        ic_per_batch.append((cov / (std_p * std_t)).mean().item())

        all_diff_pred.append(diff_pred.detach().cpu().numpy())
        all_diff_true.append(diff_true.detach().cpu().numpy())

        n_samples += output.size(0) * data.m

    _all_dp = np.vstack(all_diff_pred)
    _all_dt = np.vstack(all_diff_true)

    _move_thresh = float(np.quantile(np.abs(_all_dt), 0.50))
    _conf_thresh = float(np.quantile(np.abs(_all_dp), 0.75))

    _move_mask = np.abs(_all_dt) > _move_thresh
    _conf_mask = (np.abs(_all_dp) > _conf_thresh) & _move_mask
    _correct   = (_all_dp * _all_dt) > 0

    da  = float((_correct & _move_mask).sum()) / max(int(_move_mask.sum()), 1)
    tda = float((_correct & _conf_mask).sum()) / max(int(_conf_mask.sum()), 1)

    rse  = math.sqrt(total_loss / n_samples) / data.rse
    rae  = (total_loss_l1 / n_samples) / data.rae
    rmse = math.sqrt(total_loss / n_samples)
    mae  = total_loss_l1 / n_samples
    mape = 0.0
    ic   = float(np.mean(ic_per_batch)) if ic_per_batch else 0.0
    icir = (ic / (float(np.std(ic_per_batch)) + 1e-8)) if len(ic_per_batch) > 1 else 0.0

    predict_np = predict.data.cpu().numpy()
    Ytest_np   = test.data.cpu().numpy()
    sigma_p = predict_np.std(axis=0);  sigma_g = Ytest_np.std(axis=0)
    mean_p  = predict_np.mean(axis=0); mean_g  = Ytest_np.mean(axis=0)
    index   = sigma_g != 0
    denom   = (sigma_p * sigma_g); denom[denom < 1e-8] = 1e-8
    correlation = ((predict_np - mean_p) * (Ytest_np - mean_g)).mean(axis=0) / denom
    correlation = correlation[index].mean()

    r2_list = []
    for i in range(predict_np.shape[1]):
        y_t = Ytest_np[:, i]; y_p = predict_np[:, i]
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2_list.append(0.0 if ss_tot < 1e-5 else 1 - ss_res / ss_tot)
    r2 = np.mean(r2_list)

    if save_dir is not None and run_id is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/diff_pred_run{run_id}.npy", _all_dp)
        np.save(f"{save_dir}/diff_true_run{run_id}.npy", _all_dt)

    return rse, rae, correlation, mae, rmse, mape, r2, da, tda, ic, icir


# ══════════════════════════════════════════════════════════════════════
# 2. train()：标准训练循环（L1Loss + Adam，与 train_single_step.py 一致）
# ══════════════════════════════════════════════════════════════════════
def train(data, X, Y, LP, model, criterion, optim, batch_size):
    """
    X, Y, LP = data.train 三元素。
    主损失 MAE 在绝对价格空间（output vs Y）。
    """
    model.train()
    total_loss = 0
    n_samples  = 0
    for X_b, Y_b, LP_b in data.get_batches(X, Y, batch_size, True, lp=LP):
        model.zero_grad()
        X_in = torch.unsqueeze(X_b, dim=1).transpose(2, 3)   # (B, 1, M, P)
        output = model(X_in)
        output = torch.squeeze(output)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        scale = data.scale.expand(output.size(0), data.m)
        loss  = criterion(output * scale, Y_b * scale)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        total_loss += loss.item()
        n_samples  += output.size(0) * data.m
        optim.step()
    return total_loss / n_samples


# ══════════════════════════════════════════════════════════════════════
# 3. 模型工厂
# ══════════════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    'lstm':          LSTMBaseline,
    'tcn':           TCNBaseline,
    'agcrn':         AGCRNBaseline,
    'patchtst':      PatchTSTBaseline,
    'itransformer':  iTransformerBaseline,
}

# 各模型推荐超参（在 31 节点外汇数据上调优过的默认值）
MODEL_KWARGS = {
    'lstm':         dict(hidden_size=128, num_layers=2, dropout=0.1),
    'tcn':          dict(channels=64,    num_layers=4, kernel_size=3, dropout=0.1),
    'agcrn':        dict(hidden_dim=64,  emb_dim=10,   num_layers=2, dropout=0.1),
    'patchtst':     dict(patch_len=16,   stride=8, d_model=64, n_heads=4,
                         num_encoder_layers=3, dropout=0.1),
    'itransformer': dict(d_model=64, n_heads=4, num_layers=3, d_ff=256, dropout=0.1),
}


# ══════════════════════════════════════════════════════════════════════
# 4. main()：单次 run
# ══════════════════════════════════════════════════════════════════════
def main(run_id, args, device):
    Data = DataLoaderS(args.data, 0.6, 0.2, device,
                       args.horizon, args.seq_in_len, args.normalize)

    # 构建模型
    model_cls    = MODEL_REGISTRY[args.model]
    extra_kwargs = MODEL_KWARGS.get(args.model, {})
    model = model_cls(
        num_nodes=args.num_nodes,
        seq_in_len=args.seq_in_len,
        revin=(args.revin == 1),
        **extra_kwargs
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.model.upper()}] 参数量: {n_params:,}")

    criterion  = nn.L1Loss(reduction='sum').to(device)
    evaluateL2 = nn.MSELoss(reduction='sum').to(device)
    evaluateL1 = nn.L1Loss(reduction='sum').to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    # Baseline checkpoint 选点：纯 val_mae（公平基线）
    # 理由：Baseline 无 DirLoss 优化，DA 提升有限，用 DA 权重选点会引入噪声；
    #       统一纯 MAE 选点保证 Baseline IC 对比的公平性。
    best_score = float('inf')

    print(f'\n>>> Run {run_id+1}/{args.runs}  [{args.model.upper()}]  开始训练...')
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1], Data.train[2],
                               model, criterion, optim, args.batch_size)

            val_rse, val_rae, val_corr, val_mae, val_rmse, val_mape, \
                val_r2, val_da, val_tda, val_ic, val_icir = \
                evaluate(Data, Data.valid[0], Data.valid[1], Data.valid[2],
                         model, evaluateL2, evaluateL1, args.batch_size)

            print(f'| epoch {epoch:3d} | {time.time()-t0:5.1f}s '
                  f'| loss {train_loss:.4f} | mae {val_mae:.4f} '
                  f'| rmse {val_rmse:.4f} | da {val_da:.4f} '
                  f'| tda {val_tda:.4f} | ic {val_ic:.4f} '
                  f'| icir {val_icir:.4f}', flush=True)

            val_score = val_mae   # 纯 MAE 选点
            if val_score < best_score:
                torch.save(model.state_dict(), args.save)
                best_score = val_score

    except KeyboardInterrupt:
        print('--- Early exit ---')

    # ── 加载最佳 checkpoint，最终测试 + 保存预测 ──────────────────────
    model.load_state_dict(torch.load(args.save, map_location=device, weights_only=True))

    model_name    = os.path.splitext(os.path.basename(args.save))[0]
    save_parent   = os.path.dirname(os.path.abspath(args.save))
    # 与 train_single_step.py 路径推导逻辑完全一致：
    #   ./output/pipeline/model_LSTM.pt → ./output/pipeline/model_LSTM/
    #   ./model/model_LSTM.pt           → ./output/model_LSTM/
    if 'output' in save_parent:
        pred_save_dir = os.path.join(save_parent, model_name)
    else:
        pred_save_dir = os.path.join('./output', model_name)

    test_rse, test_rae, test_corr, test_mae, test_rmse, test_mape, \
        test_r2, test_da, test_tda, test_ic, test_icir = \
        evaluate(Data, Data.test[0], Data.test[1], Data.test[2],
                 model, evaluateL2, evaluateL1, args.batch_size,
                 save_dir=pred_save_dir, run_id=run_id)

    print(f"Run {run_id+1} Final Test [{args.model.upper()}]: "
          f"MAE {test_mae:.4f} | RMSE {test_rmse:.4f} | "
          f"DA {test_da:.4f} | tDA {test_tda:.4f} | "
          f"IC {test_ic:.4f} | ICIR {test_icir:.4f}")

    return (test_rse, test_rae, test_corr, test_mae, test_rmse,
            test_mape, test_r2, test_da, test_tda, test_ic, test_icir)


# ══════════════════════════════════════════════════════════════════════
# 5 & 6. 参数配置 + 多 run 入口（仅在直接运行时生效，import 时不执行）
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline 模型对比实验')
    parser.add_argument('--model',    type=str,   required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='baseline 模型名称')
    parser.add_argument('--data',     type=str,   default='./data/G31_RawPrice.txt')
    parser.add_argument('--num_nodes',type=int,   default=31)
    parser.add_argument('--seq_in_len',type=int,  default=24*7,  help='输入窗口长度')
    parser.add_argument('--horizon',  type=int,   default=1)
    parser.add_argument('--normalize',type=int,   default=2,
                        help='0=不归一化, 2=按最大值归一化(推荐，与 train_single_step.py 保持一致)')
    parser.add_argument('--revin',    type=int,   default=1,
                        help='1=启用 RevIN，0=禁用')
    parser.add_argument('--epochs',   type=int,   default=50)
    parser.add_argument('--runs',     type=int,   default=3)
    parser.add_argument('--batch_size',type=int,  default=128)
    parser.add_argument('--lr',       type=float, default=0.0001)
    parser.add_argument('--weight_decay',type=float,default=0.00001)
    parser.add_argument('--clip',     type=int,   default=5)
    parser.add_argument('--save',     type=str,   default='./model/model_baseline.pt')
    parser.add_argument('--device',   type=str,   default=None,
                        help='cuda / mps / cpu，不指定则自动检测')
    parser.add_argument('--metrics_tag', type=str, default='',
                        help='并行训练时指定 JSON 输出文件后缀，如 lstm → output/latest_metrics_lstm.json')
    args, _ = parser.parse_known_args()

    # ── 自动检测设备 ────────────────────────────────────────────────────
    if args.device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Using MPS (Apple GPU)')
        elif torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('Using CUDA')
        else:
            device = torch.device('cpu')
            print('Using CPU')
    else:
        device = torch.device(args.device)
        print(f'Using specified device: {device}')

    if device.type == 'cpu':
        n_cpu = os.cpu_count() or 4
        torch.set_num_threads(n_cpu)
        torch.set_num_interop_threads(max(1, n_cpu // 2))

    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)

    results = {k: [] for k in
               ['rse','rae','corr','mae','rmse','mape','r2','da','tda','ic','icir']}

    for i in range(args.runs):
        rse,rae,corr,mae,rmse,mape,r2,da,tda,ic,icir = main(i, args, device)
        for k, v in zip(results.keys(),
                        [rse,rae,corr,mae,rmse,mape,r2,da,tda,ic,icir]):
            results[k].append(v)

    print('\n' + '='*55)
    print(f'[{args.model.upper()}]  Summary over {args.runs} runs')
    print('='*55)
    print(f"{'Metric':<10} | {'Mean':<10} | {'Std':<10}")
    print('-'*38)
    for key in ['mae', 'rmse', 'da', 'tda', 'ic', 'icir']:
        mean_v = np.mean(results[key])
        std_v  = np.std(results[key])
        print(f"{key.upper():<10} | {mean_v:<10.4f} | {std_v:<10.4f}")
    print('='*55)

    # ══════════════════════════════════════════════════════════════════
    # 结构化指标导出：将本次运行的核心指标和超参数保存为 JSON
    # 供 run_final.sh 读取并追加写入 EXPERIMENT_LOG.md
    # ══════════════════════════════════════════════════════════════════
    import json, datetime
    _metrics_out = {
        "timestamp":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_save":     args.save,
        "epochs":         args.epochs,
        "runs":           args.runs,
        "batch_size":     args.batch_size,
        "use_mamba":      0,
        "dual_graph":     0,
        "use_router":     0,
        "use_dirloss":    0,
        "use_diffic":     0,
        "dir_weight":     0,
        "diffic_weight":  0,
        "phys_weight":    0,
        "ckpt_ic_weight": 0,
        "mae":       round(float(np.mean(results['mae'])),   4),
        "rmse":      round(float(np.mean(results['rmse'])),  4),
        "da":        round(float(np.mean(results['da'])),    4),
        "tda":       round(float(np.mean(results['tda'])),   4),
        "ic":        round(float(np.mean(results['ic'])),    4),
        "icir":      round(float(np.mean(results['icir'])),  4),
        "da_trend":  None,   # Baseline 无体制分析
        "da_range":  None,
        "ccc":       None,
        "da_spread": 0.0,
    }
    os.makedirs("output", exist_ok=True)
    _tag = f"_{args.metrics_tag}" if args.metrics_tag else ""
    _json_path = f"output/latest_metrics{_tag}.json"
    with open(_json_path, "w") as _f:
        json.dump(_metrics_out, _f, indent=2)
    print(f"\n[导出] 核心指标已写入 {_json_path}")
