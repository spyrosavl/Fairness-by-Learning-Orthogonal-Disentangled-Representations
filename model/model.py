import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class Tabular_ModelEncoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dims, z_dim):

        super(Tabular_ModelEncoder,self).__init__()
        
        #Shared encoding layers of the model
        self.shared_model = nn.Linear(input_dim, hidden_dims)
        
        #Output layers for each encoder
        self.mean_encoder_1 = nn.Linear(hidden_dims, z_dim)
        self.log_std_1      = nn.Linear(hidden_dims, z_dim)

        self.mean_encoder_2 = nn.Linear(hidden_dims, z_dim)
        self.log_std_2      = nn.Linear(hidden_dims, z_dim)

        #Activation function
        '(?)Not sure if in the experiments this act function is used'
        self.act_f = nn.ReLU() 
    
    def forward(self, x):
        
        #Output of shared layers followed by the activation
        out_shared = self.act_f(self.shared_model(x))
        
        #Forward of each encoder
        mean_t = self.mean_encoder_1(out_shared)
        log_std_t = self.log_std_1(out_shared)

        mean_s = self.mean_encoder_2(out_shared)
        log_std_s = self.log_std_2(out_shared)

        return mean_t, mean_s, log_std_t, log_std_s

class Tabular_ModelDecoder(nn.Module):
    
    def __init__(self, z_dim, hidden_dims, target_classes, sensitive_classes):

        super.init(Tabular_ModelDecoder, self).__init__()

        #List of layers excluding the output layer
        self.num_layers = [self.z_dim] + self.hidden_dims
        
        #Activation function
        self.act_f = nn.ReLU()

        #Architecture of the decoder
        self.layers = []
        
        for layer_index in range(1, len(self.num_layers)):
            self.layers += [nn.Linear(self.num_layers[layer_index - 1],
                            self.num_layers[layer_index]), self.act_f]
        
        #Output layer
        self.output_1 = nn.Linear(self.num_layers[-1], target_classes)
        self.output_2 = nn.Linear(self.num_layers[-1], sensitive_classes)

        self.Decoder = nn.ModuleList(self.layers)
    
    def forward(self, z_1, z_2):

        for layers in self.Decoder:
            out_1= layers(z_1)
            out_2= layers(z_2)

            x_1 = self.output_1(out_1)
            x_2 = self.output_2(out_2)

        return x_1, x_2





