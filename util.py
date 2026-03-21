import os
import numpy as np
import torch
class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2):
        self.P = window
        self.h = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = normalize
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        # RSE/RAE 基准：用测试集真实差分（(Y - LP) * scale）的标准差计算
        # test[1]=Y（归一化绝对价格），test[2]=LP（窗口最后价格）
        _scale_exp = self.scale.expand(self.test[1].size(0), self.m)
        tmp = (self.test[1] - self.test[2]) * _scale_exp   # 真实涨跌量（真实空间）

        self.scale = self.scale.to(device)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        """
        目标变量：Y[i] = dat[t+h]（归一化绝对价格）
        当前价格：LP[i] = dat[end-1]（输入窗口最后一步，即 t 时刻的归一化价格）

        设计说明：
          - Y 是绝对价格（归一化到 0-1），与 RevIN denorm 输出完全对齐
          - LP 是窗口最后一步价格，用于在 evaluate/train 中计算差分：
              diff = (Y - LP) * scale  →  真实涨跌量（单位：原始价格）
              diff_pred = (output - LP) * scale  →  预测涨跌量
          - DA / DirLoss 基于差分方向，与 MAE（绝对价格空间）天然解耦
          - 归一化与 RevIN 完全一致
        """
        n = len(idx_set)
        X  = torch.zeros((n, self.P, self.m))
        Y  = torch.zeros((n, self.m))
        LP = torch.zeros((n, self.m))   # last price in normalized space
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :]  = torch.from_numpy(self.dat[start:end, :])
            Y[i, :]     = torch.from_numpy(self.dat[idx_set[i], :])
            LP[i, :]    = torch.from_numpy(self.dat[end - 1, :])   # t 时刻价格
        return [X, Y, LP]

    def get_batches(self, inputs, targets, batch_size, shuffle=True, lp=None):
        """
        inputs: X tensor  (n, P, m)
        targets: Y tensor (n, m)
        lp: last_price tensor (n, m), 可选；若提供则 yield (X, Y, LP) 三元组
        """
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt].to(self.device)
            Y = targets[excerpt].to(self.device)
            if lp is not None:
                LP = lp[excerpt].to(self.device)
                yield X, Y, LP
            else:
                yield X, Y
            start_idx += batch_size


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


def _parse_forex_pair(name):
    """
    解析外汇货币对名称，返回 (base, quote)。
    标准格式：6 字符，前 3 为 base，后 3 为 quote。如 EURUSD -> (EUR, USD)。
    """
    if not isinstance(name, str) or len(name) < 6:
        return None
    name = name.upper()
    return (name[:3], name[3:6])


def get_forex_triangles(node_names):
    """
    遍历 node_names（货币对列表，如 EURUSD, USDJPY 等），寻找满足
    Pair(A/B) × Pair(B/C) = Pair(A/C) 三角套利逻辑的三个索引组合。

    返回: list of (i, j, k)，其中
      - 索引 i: Pair(A/B)
      - 索引 j: Pair(B/C)
      - 索引 k: Pair(A/C)
      满足 quote_i == base_j, base_i == base_k, quote_j == quote_k。
    """
    n = len(node_names)
    parsed = []
    for i, nm in enumerate(node_names):
        p = _parse_forex_pair(nm)
        parsed.append((i, p) if p else (i, None))

    triangles = []
    for i in range(n):
        base_i, quote_i = parsed[i][1] or (None, None)
        if base_i is None:
            continue
        for j in range(n):
            if i == j:
                continue
            base_j, quote_j = parsed[j][1] or (None, None)
            if base_j is None or quote_i != base_j:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                base_k, quote_k = parsed[k][1] or (None, None)
                if base_k is None or quote_j != quote_k or base_i != base_k:
                    continue
                # 找到三角: i=A/B, j=B/C, k=A/C
                triangles.append((i, j, k))
    return triangles


def get_node_names_from_data_dir(data_path):
    """
    从数据文件所在目录扫描 *-h1-bid-*.csv，提取货币对名称（与 build_dataset 顺序一致）。
    若无可用的 CSV 或目录不存在，返回 None。
    """
    import glob
    data_dir = os.path.dirname(os.path.abspath(data_path))
    pattern = os.path.join(data_dir, '*-h1-bid-*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    names = []
    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]
        pair = base.split('-')[0].upper()
        names.append(pair)
    return names

            