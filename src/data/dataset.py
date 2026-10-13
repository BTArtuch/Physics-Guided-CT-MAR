import os
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np

class MARDataset(Dataset):
    def __init__(self, data_dir, min_hu=-1000, max_hu=3000):
        """
        Expects a data directory structured as:
        data_dir/
            ├── corrupted/  (metal-corrupted CT scans)
            ├── mask/       (binary metal masks)
            └── clean/      (ground truth uncorrupted scans)
        """
        self.data_dir = data_dir
        self.min_hu = min_hu
        self.max_hu = max_hu
        
        corrupted_dir = os.path.join(data_dir, 'corrupted')
        # Assuming .nii.gz format for 3D volumes
        self.samples = [f for f in os.listdir(corrupted_dir) if f.endswith('.nii.gz')]

    def __len__(self):
        return len(self.samples)

    def normalize_hu(self, volume):
        # Clip Hounsfield Units to the standard medical window and normalize to [0, 1]
        volume = np.clip(volume, self.min_hu, self.max_hu)
        volume = (volume - self.min_hu) / (self.max_hu - self.min_hu)
        return volume

    def __getitem__(self, idx):
        filename = self.samples[idx]
        
        # Define paths
        corr_path = os.path.join(self.data_dir, 'corrupted', filename)
        mask_path = os.path.join(self.data_dir, 'mask', filename)
        clean_path = os.path.join(self.data_dir, 'clean', filename)
        
        # Load volumes using SimpleITK
        corr_vol = sitk.GetArrayFromImage(sitk.ReadImage(corr_path)).astype(np.float32)
        mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.float32)
        clean_vol = sitk.GetArrayFromImage(sitk.ReadImage(clean_path)).astype(np.float32)
        
        # Apply HU normalization
        corr_vol = self.normalize_hu(corr_vol)
        clean_vol = self.normalize_hu(clean_vol)
        
        # Add the channel dimension required by 3D U-Net: (1, Depth, Height, Width)
        corr_vol = np.expand_dims(corr_vol, axis=0)
        mask_vol = np.expand_dims(mask_vol, axis=0)
        clean_vol = np.expand_dims(clean_vol, axis=0)
        
        return {
            'corrupted': torch.from_numpy(corr_vol),
            'mask': torch.from_numpy(mask_vol),
            'clean': torch.from_numpy(clean_vol)
        }