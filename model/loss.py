import torch
import torch.nn as nn
import numpy as np

def KLD(enc_mean, enc_log_std, prior_mean, prior_cov):
    prior_log_det_cov = torch.log(torch.det(prior_cov))

    enc_var = torch.exp(2 * enc_log_std)
    enc_cov = torch.diag(enc_var).float()
    
    enc_log_det_cov = torch.log(torch.det(enc_cov))
    
    mean = prior_mean - enc_mean
    mean_trans = torch.from_numpy(np.array(prior_mean - enc_mean).T)
    
    prec_mat = torch.inverse(prior_cov)
    
    trace = torch.trace(torch.matmul(prec_mat, enc_cov))
    
    KLD = torch.sum(1 / 2 * (prior_log_det_cov - enc_log_det_cov - 2 + trace + mean_trans.unsqueeze(1) * prec_mat * mean.unsqueeze(1)))
   
    return KLD

def L_e(sen_dis_out):
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


if __name__ == "__main__":

    enc_mean = torch.from_numpy(np.array([1,2]).T)
    enc_log_std = torch.from_numpy(np.array([0.1,0.2]).T)
    prior_mean = torch.from_numpy(np.array([0,1]).T)
    prior_cov = torch.eye(2)
    kld = KLD(enc_mean, enc_log_std, prior_mean, prior_cov)
    print(kld)

    sen_dis_out = torch.from_numpy(np.array([1.2,3.3,4.1,2.2,1.1,2.1])).float()
    L_e = L_e(sen_dis_out)
    print(L_e)

    loss = loss(0.5,0.7,1,2,2000,20,torch.tensor(1.34),torch.tensor(1.54),torch.tensor(5.7),torch.tensor(10))
    print(loss)