import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTNet(nn.Module):
    
    def __init__(self, num_features=5, device='cuda'):
        super(LSTNet, self).__init__()
        self.device = device
        self.num_features = num_features
        
        # Model parameters
        self.conv_out_channels = 32
        self.gru_hidden_size = 64
        self.skip_lengths = [4, 24]  # Weekly and monthly patterns
        self.skip_hidden_size = 16
        self.ar_window = 7  # Autoregressive window size
        
        # Convolutional layer
        self.conv = nn.Conv2d(1, self.conv_out_channels, 
                             kernel_size=(7, self.num_features))
        
        # GRU layers
        self.gru = nn.GRU(self.conv_out_channels, self.gru_hidden_size, 
                         batch_first=True)
        
        # Skip GRU layers
        self.skip_gru = nn.ModuleList([
            nn.GRU(self.conv_out_channels, self.skip_hidden_size, 
                  batch_first=True)
            for _ in range(len(self.skip_lengths))
        ])
        
        # Output layers
        self.fc = nn.Linear(
            self.gru_hidden_size + 
            len(self.skip_lengths) * self.skip_hidden_size,
            self.num_features
        )
        
        # Autoregressive component
        self.ar = nn.Linear(self.ar_window, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        # Move to device
        self.to(device)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Convolutional layer
        c = x.unsqueeze(1)  # Add channel dimension
        c = F.relu(self.conv(c))
        c = self.dropout(c)
        c = c.squeeze(3)  # Remove last dimension
        
        # GRU layer
        r = c.permute(0, 2, 1)
        r, _ = self.gru(r)
        r = r[:, -1, :]
        r = self.dropout(r)
        
        # Skip connections
        skip_outputs = []
        for i, skip_len in enumerate(self.skip_lengths):
            # Get last skip_len time steps
            s = c[:, :, -skip_len:]
            
            # Process through skip GRU
            s = s.permute(0, 2, 1)
            s, _ = self.skip_gru[i](s)
            s = s[:, -1, :]
            s = self.dropout(s)
            skip_outputs.append(s)
        
        # Combine GRU and skip outputs
        combined = torch.cat([r] + skip_outputs, dim=1)
        
        # Final output
        output = self.fc(combined)
        
        # Autoregressive component
        if self.ar_window > 0:
            ar = x[:, -self.ar_window:, :]
            ar = ar.permute(0, 2, 1)
            ar = self.ar(ar)
            ar = ar.squeeze(2)
            output = output + ar
        
        return output