import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
from base import BaseModel

#@torch.no_grad()
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
#        import pdb; pdb.set_trace()
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

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}

class ResNetBlock(nn.Module):

    def __init__(self, c_in=1, act_fn=act_fn_by_name["relu"], subsample=True, c_out=-1):
        
        super().__init__()

        if not subsample:
            c_out = c_in
            
        # Network representing F
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False), # No bias needed as the Batch Norm handles it
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out))
        
        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = act_fn()
 
    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out

class PreActResNetBlock(nn.Module):

    def __init__(self, c_in=1, act_fn=act_fn_by_name["relu"], subsample=False, c_out=-1):
        
        super().__init__()

        if not subsample:
            c_out = c_in
            
        # Network representing F
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False))
        
        # 1x1 convolution needs to apply non-linearity as well as not done on skip connection
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False)) if subsample else None

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out

resnet_blocks_by_name = {
    "ResNetBlock": ResNetBlock,
    "PreActResNetBlock": PreActResNetBlock}

class ResNet(nn.Module):

    def __init__(self, num_classes=64, num_blocks=[2,2,2,2], c_hidden=[64,128,256,512], act_fn_name="relu", block_name="ResNetBlock", **kwargs):
       
        super().__init__()

        assert block_name in resnet_blocks_by_name
        self.hparams = SimpleNamespace(num_classes=num_classes, 
                                       c_hidden=c_hidden, 
                                       num_blocks=num_blocks, 
                                       act_fn_name=act_fn_name,
                                       act_fn=act_fn_by_name[act_fn_name],
                                       block_class=resnet_blocks_by_name[block_name])
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden
        
        # A first convolution on the original image to scale up the channel size
        if self.hparams.block_class == PreActResNetBlock: # => Don't apply non-linearity on output
            self.input_net = nn.Sequential(
                nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False))
        else:
            self.input_net = nn.Sequential(
                nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_hidden[0]),
                self.hparams.act_fn())
        
        # Creating the ResNet blocks
        blocks = []
        for block_idx, block_count in enumerate(self.hparams.num_blocks):
            for bc in range(block_count):
                subsample = (bc == 0 and block_idx > 0) # Subsample the first block of each group, except the very first one.
                blocks.append(
                    self.hparams.block_class(c_in=c_hidden[block_idx if not subsample else (block_idx-1)],
                                             act_fn=self.hparams.act_fn,
                                             subsample=subsample,
                                             c_out=c_hidden[block_idx]))

        self.blocks = nn.Sequential(*blocks)
        
        # Mapping to output
        self.output1_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], self.hparams.num_classes))

        self.output2_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], self.hparams.num_classes))
        
    def _init_params(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x_1 = self.output1_net(x)
        x_2 = self.output2_net(x)
        return x_1, x_2

class CIFAR_Encoder(nn.Module):

    def __init__(self, input_dim, z_dim):

        super(CIFAR_Encoder, self).__init__()
        
        self.z_dim = z_dim

        #Output layers for each encoder
        self.mean_encoder_1 = nn.Linear(input_dim, z_dim)
        self.log_std_1      = nn.Linear(input_dim, z_dim)

        self.mean_encoder_2 = nn.Linear(input_dim, z_dim)
        self.log_std_2      = nn.Linear(input_dim, z_dim)

        #Activation function
        self.act_f = nn.ReLU() 

    def forward(self, x):

        out_1, out_2 = self.act_f(ResNet(self.z_dim).forward(x)[0]), self.act_f(ResNet(self.z_dim).forward(x)[1])
        
        mean_t = self.mean_encoder_1(torch.transpose(out_1, 0, 1))
        log_std_t = self.log_std_1(torch.transpose(out_1, 0, 1))
        
        mean_s = self.mean_encoder_2(torch.transpose(out_2, 0, 1))
        log_std_s = self.log_std_2(torch.transpose(out_2, 0, 1))

        return mean_t, mean_s, log_std_t, log_std_s

class CifarModel(BaseModel):

    def __init__(self, input_dim, hidden_dim, z_dim, target_classes, sensitive_classes):
        super().__init__()

        self.encoder = CIFAR_Encoder(input_dim, z_dim)
        self.decoder = Tabular_ModelDecoder(z_dim, [hidden_dim, hidden_dim], target_classes, sensitive_classes)

    def forward(self, x):
        mean_t, mean_s, log_std_t, log_std_s = self.encoder(x)
        z1, z2 = reparameterization(mean_t, mean_s, log_std_t, log_std_s)
        y_zt, s_zt, s_zs = self.decoder(z1, z2) 
        return (mean_t, mean_s, log_std_t, log_std_s), (y_zt, s_zt, s_zs), (z1, z2)
        
