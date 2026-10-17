import torch
from projector.autograd import RadonTransform

def test_point_projection():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create an empty volume [Batch, Channels, Depth, Height, Width]
    volume = torch.zeros(1, 1, 16, 32, 32, device=device)
    
    # Insert a single high-intensity "metal" voxel off-center on the U-axis
    # volume[0, 0, 8, 16, 24] = 1000.0
    volume[0, 0, 8, 16, 18] = 1000.0 
    
    # Define 4 orthogonal projection angles (0, 90, 180, 270 degrees)
    angles = torch.tensor([0.0, 1.5708, 3.1416, 4.7124], device=device)
    
    # Forward project using the custom CUDA kernel
    radon = RadonTransform(angles)
    sinogram = radon(volume)
    
    print("Physical Geometry Point Test:")
    for i, angle in enumerate(angles):
        # Sum across the vertical detector (V) to see horizontal displacement (U)
        det_u_profile = sinogram[0, 0, i].sum(dim=0)
        
        max_u_idx = torch.argmax(det_u_profile).item()
        max_val = det_u_profile[max_u_idx].item()
        
        angle_deg = int(angle.item() * 180 / 3.14159)
        print(f"Angle {angle_deg:3d}° -> Peak X-ray shadow at Detector U-index: {max_u_idx:2d} (Intensity: {max_val:.1f})")

if __name__ == "__main__":
    test_point_projection()