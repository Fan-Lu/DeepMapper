

import torch
import scipy
from scipy.ndimage import uniform_filter1d
from statsmodels.nonparametric.kernel_regression import KernelReg

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


