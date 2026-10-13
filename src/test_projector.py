import torch
from models.unet3d import UNet3D
from models.loss import DualDomainLoss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Dummy inputs: [Batch, Channels (CT + Metal Mask), Depth, Height, Width]
    corrupted_ct = torch.randn(1, 1, 64, 128, 128, device=device)
    metal_mask = torch.randint(0, 2, (1, 1, 64, 128, 128), device=device).float()
    target_ct = torch.randn(1, 1, 64, 128, 128, device=device)

    # 2. Dummy projection-space data: [Batch, Channels, Angles, Det_V, Det_U]
    angles = torch.linspace(0, 3.14159, 180, device=device)
    measured_sino = torch.randn(1, 1, 180, 64, 128, device=device)
    valid_sino_mask = torch.ones_like(measured_sino)

    # 3. Initialize 3D U-Net and Dual-Domain Loss
    model = UNet3D(in_channels=2, out_channels=1, base_features=16).to(device)
    criterion = DualDomainLoss(projection_angles=angles, lambda_radon=0.1).to(device)

    # 4. Forward & Backward Pass
    model_input = torch.cat([corrupted_ct, metal_mask], dim=1)
    pred_volume = model(model_input)

    total_loss, loss_img, loss_radon = criterion(
        pred_volume, target_ct, measured_sino, valid_sino_mask
    )
    total_loss.backward()

    print(f"Predicted Volume Shape : {pred_volume.shape}")
    print(f"Total Loss             : {total_loss.item():.4f} (Image: {loss_img.item():.4f}, Radon: {loss_radon.item():.4f})")

if __name__ == "__main__":
    main()