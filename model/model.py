import torch
import torch.nn as nn
import numpy as np
from base import BaseModel
#from utils.util import reparameterization

def reparameterization(mean_t, mean_s, log_std_t, log_std_s):
    z1 = mean_t + torch.exp(log_std_t - torch.max(log_std_t)) @ torch.normal(torch.from_numpy(np.array([0.,1.]).T).float(), torch.eye(2))
    z2 = mean_s + torch.exp(log_std_s - torch.max(log_std_s)) @ torch.normal(torch.from_numpy(np.array([1.,0.]).T).float(), torch.eye(2))
    return z1, z2

class TabularModel(BaseModel):

    def __init__(self, input_dim, hidden_dim, z_dim, target_classes, sensitive_classes):
        super(TabularModel, self).__init__()

        self.encoder = Tabular_ModelEncoder(input_dim, hidden_dim, z_dim)
        self.decoder = Tabular_ModelDecoder(z_dim, [hidden_dim, hidden_dim], target_classes, sensitive_classes)


    def forward(self, x):
        #import pdb; pdb.set_trace()
        mean_t, mean_s, log_std_t, log_std_s = self.encoder(x)
        z1, z2 = reparameterization(mean_t, mean_s, log_std_t, log_std_s)
        y_zt, s_zt, s_zs = self.decoder(z1, z2) 
        return (mean_t, mean_s, log_std_t, log_std_s), (y_zt, s_zt, s_zs), (z1, z2)


class Tabular_ModelEncoder(BaseModel):
    
    def __init__(self, input_dim, hidden_dim, z_dim):

        super(Tabular_ModelEncoder,self).__init__()
        
        #Shared encoding layers of the model
        self.shared_model = nn.Linear(input_dim, hidden_dim)
        
        #Different encoding layers
        self.encoder_1 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_2 = nn.Linear(hidden_dim, hidden_dim)

        #Output layers for each encoder
        self.mean_encoder_1 = nn.Linear(hidden_dim, z_dim)
        self.log_std_1      = nn.Linear(hidden_dim, z_dim)

        self.mean_encoder_2 = nn.Linear(hidden_dim, z_dim)
        self.log_std_2      = nn.Linear(hidden_dim, z_dim)

        #Activation function
        self.act_f = nn.ReLU() 
    
    def forward(self, x):
        x = x.float()
        #Output of shared layers followed by the activation
        out_shared = self.act_f(self.shared_model(x))
        
        #Forward of each encoder
        out_1 = self.act_f(self.encoder_1(out_shared))
        mean_t = self.mean_encoder_1(out_1)
        log_std_t = self.log_std_1(out_1)
        
        out_2 = self.act_f(self.encoder_2(out_shared))
        mean_s = self.mean_encoder_2(out_2)
        log_std_s = self.log_std_2(out_2)

        return mean_t, mean_s, log_std_t, log_std_s



class Tabular_ModelDecoder(BaseModel):
    
    def __init__(self, z_dim, hidden_dims, target_classes, sensitive_classes):

        super(Tabular_ModelDecoder, self).__init__()

        #List of layers excluding the output layer
        self.z_dim = z_dim
        self.hidden_dims = hidden_dims
        self.target_classes = target_classes
        self.sensitive_classes = sensitive_classes

        #import pdb; pdb.set_trace()
        self.num_layers = [self.z_dim] + self.hidden_dims
        
        #Activation function
        self.act_f = nn.ReLU()

        #Architecture of the decoder 1
        self.layers_1 = []
        for layer_index_1 in range(1, len(self.num_layers)):
            self.layers_1 += [nn.Linear(self.num_layers[layer_index_1 - 1],
                            self.num_layers[layer_index_1]), self.act_f]
        
        #Architecture of the decoder 2
        self.layers_2 = []
        for layer_index_2 in range(1, len(self.num_layers)):
            self.layers_2 += [nn.Linear(self.num_layers[layer_index_2 - 1],
                            self.num_layers[layer_index_2]), self.act_f]
        
        #Output layer
        self.output_1 = nn.Linear(self.num_layers[-1], self.target_classes)
        self.output_2 = nn.Linear(self.num_layers[-1], self.sensitive_classes)
        
        self.Decoder_1 = nn.ModuleList(self.layers_1)
        self.Decoder_2 = nn.ModuleList(self.layers_2)
    
    def forward(self, z_1, z_2):

        out_1 = z_1
        for layers_1 in self.Decoder_1:
            out_1 = layers_1(out_1)
        y_zt = self.output_1(out_1)
            
        out_1 = z_1
        out_2 = z_2
        for layers_2 in self.Decoder_2:
            out_1 = layers_2(out_1)
            out_2 = layers_2(out_2)
        s_zt = self.output_2(out_1)
        s_zs = self.output_2(out_2)

        return y_zt, s_zt, s_zs
