import json
import torch

import pandas as pd
import torch.nn as nn

from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from model.model import *
from model.loss import *
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal


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


class Criterion(nn.Module):
    def __init__(self, lambda_e, lambda_od, gamma_e, gamma_od, step_size):
        super(Criterion, self).__init__()
        self.lambda_e = lambda_e
        self.lambda_od = lambda_od
        self.gamma_e = gamma_e
        self.gamma_od = gamma_od
        self.step_size = step_size

        self.bce = nn.BCEWithLogitsLoss()
        self.cross = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss(reduction='batchmean')
        #TODO tensors through which we have to back propagate have to have require_grad = True (and be of type Parameter?)

    def forward(self, inputs, target, sensitive, dataset_name, current_step):
        mean_t, mean_s, log_std_t, log_std_s = inputs[0]
        y_zt, s_zt, s_zs = inputs[1]
        z1, z2 = inputs[2]

        if dataset_name in ['CIFAR10DataLoader', 'CIFAR100DataLoader']:
            L_t = self.cross(y_zt, target)
            mean_1, mean_2 = mean_tensors(np.zeros(128), np.ones(128), 13)
            m_t = MultivariateNormal(mean_1, torch.eye(128))
            m_s = MultivariateNormal(mean_2, torch.eye(128))
        elif dataset_name == 'YaleDataLoader':
            target_arg_max = target.argmax(dim=1)
            L_t = self.cross(y_zt, target_arg_max)
            mean_1, mean_2 = mean_tensors(np.zeros(100), np.ones(100), 13)
            m_t = MultivariateNormal(mean_1, torch.eye(100))
            m_s = MultivariateNormal(mean_2, torch.eye(100)) 
        else: #tabular
            L_t = self.bce(y_zt, target[:,None].float())
            m_t = MultivariateNormal(torch.tensor([0.,1.]), torch.eye(2))
            m_s = MultivariateNormal(torch.tensor([1.,0.]), torch.eye(2))
        
        #uniform = torch.rand(size=s_zt.size())
        #Loss_e = self.kld(torch.log_softmax(s_zt, dim=1), torch.softmax(uniform, dim=1))
        Loss_e = L_e(s_zt)
        
        #TODO should the priors be the same for each loss computation?
        # --> should we define them in init?       
        prior_t=[]; prior_s=[]
        enc_dis_t=[]; enc_dis_s=[]

        try:
            for i in range(z1.shape[0]):
                prior_t.append(m_t.sample())
                prior_s.append(m_s.sample())
                n_t = MultivariateNormal(mean_t[i], torch.diag(torch.exp(log_std_t[i])))
                n_s = MultivariateNormal(mean_s[i], torch.diag(torch.exp(log_std_s[i])))
                enc_dis_t.append(n_t.sample())
                enc_dis_s.append(n_s.sample())
        except:
            import pdb; pdb.set_trace()

        prior_t = torch.stack(prior_t)
        prior_s = torch.stack(prior_s)
        enc_dis_t = torch.stack(enc_dis_t)
        enc_dis_s = torch.stack(enc_dis_s)
        
        #print(enc_dis_s)
       
        #print(torch.softmax(enc_dis_t, dim=1))
        L_zt = self.kld(torch.log_softmax(prior_t, dim=1), torch.softmax(enc_dis_t, dim=1))
        L_zs = self.kld(torch.log_softmax(prior_s, dim=1), torch.softmax(enc_dis_s, dim=1))

        #print(L_zs, L_zt)
        lambda_e = self.lambda_e * self.gamma_e ** (current_step/self.step_size)
        lambda_od = self.lambda_od * self.gamma_od ** (current_step/self.step_size)
        Loss = L_t + lambda_e * Loss_e + lambda_od * (L_zt + L_zs)
     
        return Loss



