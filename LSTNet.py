import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTNet(nn.Module):
    
    def __init__(self, device='cuda'):
        super(LSTNet, self).__init__()
        self.device = device
        self.num_features = 5
        self.conv1_out_channels = 32 
        self.conv1_kernel_height = 7
        self.recc1_out_channels = 64 
        self.skip_steps = [4, 24] 
        self.skip_reccs_out_channels = [4, 4] 
        self.output_out_features = 5
        self.ar_window_size = 7
        self.dropout = nn.Dropout(p = 0.2)
       
        # Initialize layers
        self.conv1 = nn.Conv2d(1, self.conv1_out_channels, 
                               kernel_size=(self.conv1_kernel_height, self.num_features))
        self.recc1 = nn.GRU(self.conv1_out_channels, self.recc1_out_channels, batch_first=True)
        
        # Initialize skip recurrent layers
        self.skip_reccs = nn.ModuleList([
            nn.GRU(self.conv1_out_channels, self.skip_reccs_out_channels[i], batch_first=True)
            for i in range(len(self.skip_steps))
        ])
        
        self.output_in_features = self.recc1_out_channels + np.dot(self.skip_steps, self.skip_reccs_out_channels)
        self.output = nn.Linear(self.output_in_features, self.output_out_features)
        if self.ar_window_size > 0:
            self.ar = nn.Linear(self.ar_window_size, 1)
        
        # Move all parameters to device
        self.to(device)
        
    def forward(self, X):
        """
        Parameters:
        X (tensor) [batch_size, time_steps, num_features]
        """
        batch_size = X.size(0)
        time_steps = X.size(1)
        
        # Convolutional Layer
        C = X.unsqueeze(1) # [batch_size, num_channels=1, time_steps, num_features]
        C = F.relu(self.conv1(C)) # [batch_size, conv1_out_channels, shrinked_time_steps, 1]
        C = self.dropout(C)
        C = torch.squeeze(C, 3) # [batch_size, conv1_out_channels, shrinked_time_steps]
        
        # Recurrent Layer
        R = C.permute(0, 2, 1) # [batch_size, shrinked_time_steps, conv1_out_channels]
        out, hidden = self.recc1(R) # [batch_size, shrinked_time_steps, recc_out_channels]
        R = out[:, -1, :] # [batch_size, recc_out_channels]
        R = self.dropout(R)
        
        # Skip Recurrent Layers
        shrinked_time_steps = C.size(2)
        for i in range(len(self.skip_steps)):
            skip_step = self.skip_steps[i]
            skip_sequence_len = max(1, shrinked_time_steps // skip_step)
            
            # Get the last skip_sequence_len*skip_step time steps
            S = C[:, :, -skip_sequence_len*skip_step:] # [batch_size, conv1_out_channels, skip_sequence_len*skip_step]
            
            # Reshape for skip connections
            S = S.view(batch_size, self.conv1_out_channels, skip_sequence_len, skip_step)
            S = S.permute(0, 2, 1, 3).contiguous() # [batch_size, skip_sequence_len, conv1_out_channels, skip_step]
            S = S.view(batch_size*skip_sequence_len, self.conv1_out_channels, skip_step)
            S = S.permute(0, 2, 1).contiguous() # [batch_size*skip_sequence_len, skip_step, conv1_out_channels]
            
            # Process through GRU
            out, hidden = self.skip_reccs[i](S) # [batch_size*skip_sequence_len, skip_step, skip_reccs_out_channels[i]]
            S = out[:, -1, :] # [batch_size*skip_sequence_len, skip_reccs_out_channels[i]]
            S = S.view(batch_size, skip_sequence_len*self.skip_reccs_out_channels[i])
            S = self.dropout(S)
            R = torch.cat((R, S), 1)
        
        # Output Layer
        O = F.relu(self.output(R)) # [batch_size, output_out_features]
        
        if self.ar_window_size > 0:
            AR = X[:, -self.ar_window_size:, :] # [batch_size, ar_window_size, num_features]
            AR = AR.permute(0, 2, 1).contiguous() # [batch_size, num_features, ar_window_size]
            AR = self.ar(AR) # [batch_size, num_features, 1]
            AR = AR.squeeze(2) # [batch_size, num_features]
            O = O + AR
        
        return O