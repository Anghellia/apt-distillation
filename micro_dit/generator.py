import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class APTGenerator(nn.Module):
    def __init__(self, dit_model, final_timestep: int = 1000, use_checkpoint: bool = False):
        super().__init__()

        self.model = dit_model
        self.final_timestep = final_timestep

    def forward(self, z, context, seq_len):
        # Always use final timestep for one-step generation
        batch_size = z.shape[0]
        t = torch.ones(batch_size,  device=z.device) * self.final_timestep

        # Predict velocity field
        v = self.model(z, t, context, seq_len)

        # Convert to sample
        x = z - v

        return x
