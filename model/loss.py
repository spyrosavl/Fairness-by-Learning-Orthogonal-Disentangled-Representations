import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def KLD(enc_mean, enc_log_std, prior_mean, prior_cov):

    prior_log_det_cov = torch.log(torch.det(prior_cov))

    enc_var = torch.exp(2 * enc_log_std)
    
    for i in range(enc_log_std.shape[0]):
        enc_cov = torch.diag(enc_var[i]).float()
    
        enc_log_det_cov = torch.log(torch.det(enc_cov))

        mean = prior_mean - enc_mean[i]

        mean_trans = torch.from_numpy(np.array(prior_mean.detach() - enc_mean[i].detach()).T)

        prec_mat = torch.inverse(prior_cov)

        trace = torch.trace(torch.matmul(prec_mat, enc_cov))

    KLD = torch.sum(1 / 2 * (prior_log_det_cov - enc_log_det_cov - 2 + trace + mean_trans * prec_mat * mean))
    
    return KLD

def L_e(sen_dis_out):
#    import pdb; pdb.set_trace()
    L_e = torch.sum(sen_dis_out * torch.log(sen_dis_out))
    return L_e

def L_t(tar_cond, tar_disc_out):
    L_t = - torch.sum(tar_cond * torch.log(tar_disc_out))
    return L_t

def L_s(sen_cond, sen_disc_out):
    L_s = - torch.sum(sen_cond * torch.log(sen_disc_out))
    return L_s

def L_od(L_zt, L_zs):
    L_od = L_zt + L_zs
    return L_od

def loss(lamda_od, lamda_e, gamma_od, gamma_e, steps, current_step, L_s, L_t, L_e, L_od):
    loss = (L_t + L_s + lamda_e * gamma_e ** (current_step / steps) 
           * L_e + lamda_od * gamma_od ** (current_step / steps) * L_od)
    return loss
