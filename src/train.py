import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MARDataset
from models.unet3d import UNet3D
from models.loss import DualDomainLoss

def train():
    # 1. Hyperparameters & Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 50
    batch_size = 1
    learning_rate = 1e-4
    
    # Projection settings for the Radon Transform Penalty
    num_angles = 180
    angles = torch.linspace(0, 3.14159, num_angles, device=device)
    
    # 2. Data Loading
    # TODO: Replace with actual dataset path
    dataset = MARDataset(data_dir='./dataset_path')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 3. Model & Loss Initialization
    model = UNet3D(in_channels=2, out_channels=1, base_features=32).to(device)
    criterion = DualDomainLoss(projection_angles=angles, lambda_radon=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Move data to GPU
            corrupted_ct = batch['corrupted'].to(device)
            metal_mask = batch['mask'].to(device)
            clean_ct = batch['clean'].to(device)
            
            # Combine corrupted CT and metal mask as the 2-channel input
            model_input = torch.cat([corrupted_ct, metal_mask], dim=1)
            
            # Forward pass
            optimizer.zero_grad()
            pred_volume = model(model_input)
            
            # Note: For real training, measured_sinogram should be loaded from the dataset 
            # or synthesized from the clean_ct. Here we synthesize it from the ground truth.
            with torch.no_grad():
                measured_sinogram = criterion.radon_transform(clean_ct)
                valid_sinogram_mask = torch.ones_like(measured_sinogram) # Update logic for actual metal trace masking
            
            # Compute Hybrid Loss
            loss, loss_img, loss_radon = criterion(
                pred_volume, clean_ct, measured_sinogram, valid_sinogram_mask
            )
            
            # Backward pass & optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Img': f"{loss_img.item():.4f}", 'Radon': f"{loss_radon.item():.4f}"})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"mar_unet3d_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    train()