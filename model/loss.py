import torch
import torch.nn as nn

def KLD(enc_mean, enc_log_std, prior_mean, prior_std):
    prior_log_std = torch.log(prior_std)
    enc_var   = torch.square(torch.exp(enc_log_std))
    prior_var = torch.square(prior_std)
    mean_sq = torch.square(enc_mean - prior_mean)
    KLD = prior_log_std - enc_log_std + (enc_var + mean_sq) 
    / 2 * prior_var - 1 / 2
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

