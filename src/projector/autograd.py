import torch
import torch.nn as nn
import mar_projector_backend as backend

class JosephForwardProjector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, angles):
        # Save angles for the backward pass and record the original volume shape
        ctx.save_for_backward(angles)
        ctx.volume_shape = volume.shape
        
        # Enforce contiguous memory layout for the C++ backend
        volume = volume.contiguous()
        angles = angles.contiguous()
        
        # Execute the custom CUDA forward kernel
        sinogram = backend.forward(volume, angles)
        return sinogram

    @staticmethod
    def backward(ctx, grad_sinogram):
        # Retrieve saved context
        angles, = ctx.saved_tensors
        volume_shape = ctx.volume_shape
        grad_sinogram = grad_sinogram.contiguous()
        
        # Execute the custom CUDA backward kernel to get volume gradients
        grad_volume = backend.backward(grad_sinogram, angles, list(volume_shape))
        
        # Return gradients corresponding to the input arguments of forward()
        # (grad_volume matches 'volume', None matches 'angles' as we don't optimize angles)
        return grad_volume, None

class RadonTransform(nn.Module):
    """
    A standard PyTorch nn.Module wrapper for the custom autograd function.
    """
    def __init__(self, angles):
        super().__init__()
        # Register angles as a buffer so it moves to the correct device with the model
        self.register_buffer('angles', angles)
        
    def forward(self, volume):
        return JosephForwardProjector.apply(volume, self.angles)