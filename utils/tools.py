

import torch
from statsmodels.nonparametric.kernel_regression import KernelReg
import numpy as np
import pandas as pd
import os

def readPercentileSingle(data):

    n = len(data[0])

    def stat_rewards(data, stat):
        return [stat([e[i] for e in data]) for i in range(n)]

    data_median = stat_rewards(data, lambda l: np.percentile(l, 50))
    data_90 = stat_rewards(data, lambda l: np.percentile(l, 90))
    data_75 = stat_rewards(data, lambda l: np.percentile(l, 75))
    data_25 = stat_rewards(data, lambda l: np.percentile(l, 25))
    data_10 = stat_rewards(data, lambda l: np.percentile(l, 10))

    return data_median, data_10, data_25, data_75, data_90

def readPercentileMulti(data):
    d = len(data[0][0])

    data_medians, data_10s, data_25s, data_75s, data_90s = [], [], [], [], []

    for i in range(d):

        dtmp = [[data[s][t][i] for t in range(len(data[0]))] for s in range(len(data))]
        data_median, data_10, data_25, data_75, data_90 = readPercentileSingle(dtmp)
        data_medians.append(data_median)
        data_10s.append(data_10)
        data_25s.append(data_25)
        data_75s.append(data_75)
        data_90s.append(data_90)

    return data_medians, data_10s, data_25s, data_75s, data_90s

def read_all_seed_data(data_dir):

    dt_all_seeds = []

    im_names = os.listdir(data_dir)
    for imtmp in im_names:
        if not imtmp.endswith('.csv'):
            continue
        dt_all_seeds.append(
            pd.read_csv(os.path.join(data_dir, imtmp)).values[:, -4:]
        )
    return dt_all_seeds

def str2sec(t):
    month_days = {1: 31, 2: 29, 3: 31, 4: 30,
                  5: 31, 6: 30, 7: 31, 8: 31,
                  9: 30, 10: 31, 11: 30, 12: 31}
    ftr = [86400.0, 3600.0, 60.0]
    year, month, day, hour, min = t.split('-')
    total_time = sum([a * b for a, b in zip(ftr, [int(day), int(hour), int(min)])]) + month_days[int(month)] * 86400.0
    return total_time


def smoother(y):
    kr = KernelReg(y, range(len(y)), 'c')
    y_pred, y_std = kr.fit(range(len(y)))
    return y_pred

class SharedAdam(torch.optim.Adam):
    """
    share adam optimizer between workers
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


