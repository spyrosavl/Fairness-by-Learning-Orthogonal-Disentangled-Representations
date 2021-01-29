import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
from model.model import *

def L_e(sen_dis_out):
    L_e = -torch.sum(torch.softmax(sen_dis_out, dim=1) * torch.log_softmax(sen_dis_out, dim=1)) / sen_dis_out.shape[0]
    return L_e

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
        
        Loss_e = L_e(s_zt)
        prior_t=[]; prior_s=[]
        enc_dis_t=[]; enc_dis_s=[]

        for i in range(z1.shape[0]):
            prior_t.append(m_t.sample())
            prior_s.append(m_s.sample())
            n_t = MultivariateNormal(mean_t[i], torch.diag(torch.exp(log_std_t[i])))
            n_s = MultivariateNormal(mean_s[i], torch.diag(torch.exp(log_std_s[i])))
            enc_dis_t.append(n_t.sample())
            enc_dis_s.append(n_s.sample())
    
        prior_t = torch.stack(prior_t)
        prior_s = torch.stack(prior_s)
        enc_dis_t = torch.stack(enc_dis_t)
        enc_dis_s = torch.stack(enc_dis_s)
        
        L_zt = self.kld(torch.log_softmax(prior_t, dim=1), torch.softmax(enc_dis_t, dim=1))
        L_zs = self.kld(torch.log_softmax(prior_s, dim=1), torch.softmax(enc_dis_s, dim=1))
        
        lambda_e = self.lambda_e * self.gamma_e ** (current_step/self.step_size)
        lambda_od = self.lambda_od * self.gamma_od ** (current_step/self.step_size)
        Loss = L_t + lambda_e * Loss_e + lambda_od * (L_zt + L_zs)
        
        return Loss