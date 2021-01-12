import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from model.model import *
from model.loss import *


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def reparameterization(mean_t, mean_s, log_std_t, log_std_s):
    z1 = mean_t + torch.exp(log_std_t) * torch.normal(torch.from_numpy(np.array([0,1]).T), torch.eye(2))
    z2 = mean_s + torch.exp(log_std_s) * torch.normal(torch.from_numpy(np.array([1,0]).T), torch.eye(2))
    return z1,z2

def loss_forward(data_input):
    mean_t, mean_s, log_std_t, log_std_s = Tabular_ModelEncoder().forward(data_input)

    prior_mean_t = torch.from_numpy(np.array([0,1]).T)
    prior_std_t = 'If the covariance is the identity what is the prior std?'
    prior_mean_s = torch.from_numpy(np.array([1,0]).T)
    prior_std_s = 'If the covariance is the identity what is the prior std?'

    L_zt = KLD(mean_t, log_std_t, prior_mean_t, prior_std_t)
    L_zs = KLD(mean_s, log_std_s, prior_mean_s, prior_std_s)
    L_od = L_od(L_zt,L_zs)

    z_1, z_2 = reparameterization(mean_t, mean_s, log_std_t, log_std_s)

    y_zt, s_zt, s_zs = Tabular_ModelDecoder().forward(z_1, z_2)
    
    tar_cond = 'Not sure how these are implemented.It represents p(y|x)'
    sen_cond = 'Not sure how these are implemented.It represents p(s|x)'

    L_t = L_t(tar_cond, y_zt)
    L_s = L_s(sen_cond, s_zs)
    L_e = L_e(s_zt)

    return L_od, L_t, L_s, L_e




