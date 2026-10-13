import torch
import torch.nn as nn
from projector.autograd import RadonTransform

class DualDomainLoss(nn.Module):
    def __init__(self, projection_angles, lambda_radon=0.1):
        super().__init__()
        self.radon_transform = RadonTransform(projection_angles)
        self.lambda_radon = lambda_radon
        
        # Standard restoration metrics
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_volume, target_volume, measured_sinogram, valid_sinogram_mask):
        # 1. Image Domain Loss (L1 distance to ground-truth clean scan)
        loss_image = self.l1_loss(pred_volume, target_volume)
        
        # 2. Projection Domain Loss (Physics Consistency)
        # Synthesize the sinogram from the network's predicted 3D volume
        synth_sinogram = self.radon_transform(pred_volume)
        
        # Mask out the metal traces to compare only the valid, uncorrupted rays
        valid_synth = synth_sinogram * valid_sinogram_mask
        valid_measured = measured_sinogram * valid_sinogram_mask
        
        # Compute MSE between the valid synthesized rays and original detector measurements
        loss_radon = self.mse_loss(valid_synth, valid_measured)
        
        # 3. Total Hybrid Loss
        total_loss = loss_image + (self.lambda_radon * loss_radon)
        
        return total_loss, loss_image, loss_radon