import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor

class Encoder(nn.Module):
    def __init__(self, input_size, num_units1=128, num_units2=64, latent_size=3):
        super().__init__()
        self.input_size = input_size
        self.num_units1 = num_units1
        self.num_units2 = num_units2
        self.latent_size = latent_size
        
        self.encode = nn.Sequential(
            nn.Linear(self.input_size, self.num_units1),
            nn.ReLU(),
            nn.Linear(self.num_units1, self.num_units2),
            nn.ReLU(),
            nn.Linear(self.num_units2, self.latent_size),
            nn.ReLU(),
        )
        
    def forward(self, X):
        encoded = self.encode(X)
        return encoded
    
class Decoder(nn.Module):
    def __init__(self, input_size, num_units1=128, num_units2=64, latent_size=3):
        super().__init__()
        self.input_size = input_size
        self.num_units1 = num_units1
        self.num_units2 = num_units2
        self.latent_size = latent_size
        
        self.decode = nn.Sequential(
            nn.Linear(self.latent_size, self.num_units2),
            nn.ReLU(),
            nn.Linear(self.num_units2, self.num_units1),
            nn.ReLU(),
            nn.Linear(self.num_units1, self.input_size),
        )
        
    def forward(self, X):
        decoded = self.decode(X)
        return decoded
    
class AutoEncoder(nn.Module):
    def __init__(self, input_size, num_units1=128, num_units2=64, latent_size=3):
        super().__init__()
        self.input_size = input_size
        self.num_units1 = num_units1
        self.num_units2 = num_units2
        self.latent_size = latent_size

        self.encoder = Encoder(
            input_size=self.input_size, 
            num_units1=self.num_units1, 
            num_units2=self.num_units2,
            latent_size=self.latent_size,
            )
        self.decoder = Decoder(
            input_size=self.input_size, 
            num_units1=self.num_units1, 
            num_units2=self.num_units2,
            latent_size=self.latent_size,
            )
        
    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
class AutoEncoderNet(NeuralNetRegressor):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        decoded, encoded = y_pred
        loss_reconstruction = super().get_loss(decoded, y_true, *args, **kwargs)
        loss_l1 = 1e-3 * torch.abs(encoded).sum()
        return loss_reconstruction + loss_l1