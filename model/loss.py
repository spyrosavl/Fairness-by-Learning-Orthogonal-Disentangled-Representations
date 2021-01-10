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

