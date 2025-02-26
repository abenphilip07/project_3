import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from pathlib import Path
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Tuple, List, Optional

class FCMT(nn.Module):
    def __init__(self, input_dim: int = 3072, num_heads: int = 8, 
                 num_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.max_patches = 36  # Fixed number of patches
        
        # Improved dimension reduction with layer normalization
        self.dim_reduce = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Optimized item-level map learning
        self.H0 = nn.Sequential(
            nn.Conv1d(1024, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=1)
        )
        
        # Enhanced transformer with dropout and layer norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm architecture for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(1024)
        
        # Learnable temperature parameter for attention
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_patches, feat_dim = x.shape
        
        # Handle variable patch numbers efficiently
        if num_patches != self.max_patches:
            x = self._adjust_patches(x, batch_size, feat_dim)
        
        # Apply dimension reduction
        x = self.dim_reduce(x)  # [B, P, 1024]
        
        # Compute attention maps more efficiently
        features = x.transpose(1, 2)  # [B, 1024, P]
        attention_maps = self._compute_attention_maps(features)
        
        # Apply transformer with attention mask for padded sequences
        mask = self._create_padding_mask(num_patches) if num_patches < self.max_patches else None
        x = self.transformer(x, src_key_padding_mask=mask)
        
        return self.layer_norm(x), attention_maps

    def _adjust_patches(self, x: torch.Tensor, batch_size: int, feat_dim: int) -> torch.Tensor:
        if x.shape[1] < self.max_patches:
            padding = torch.zeros(batch_size, self.max_patches - x.shape[1], feat_dim, 
                                device=x.device, dtype=x.dtype)
            return torch.cat([x, padding], dim=1)
        return x[:, :self.max_patches, :]

    def _create_padding_mask(self, num_patches: int) -> torch.Tensor:
        mask = torch.zeros(self.max_patches, dtype=torch.bool)
        mask[num_patches:] = True
        return mask

    def _compute_attention_maps(self, features: torch.Tensor) -> torch.Tensor:
        B, C, P = features.shape
        maps = []
        
        # Compute all maps in parallel
        for i in range(self.max_patches):
            other_features = torch.cat([features[:, :, :i], features[:, :, i+1:]], dim=2)
            map_i = torch.sigmoid(self.H0(other_features).squeeze(1) / self.temperature)
            maps.append(map_i)
            
        return torch.stack(maps, dim=1)

class MultiScaleSpatialRepresentation(nn.Module):
    def __init__(self, input_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        
        # Multi-scale processing
        self.transforms = nn.ModuleDict({
            'ss': transforms.Resize((input_size[0]//2, input_size[1]//2)),
            'ms': nn.Identity(),
            'ls': transforms.Resize((input_size[0]*2, input_size[1]*2))
        })
        
        # FCMTs for different scales
        patch_dim = 32 * 32 * 3
        self.fcmts = nn.ModuleDict({
            'ss': FCMT(input_dim=patch_dim),
            'ms': FCMT(input_dim=patch_dim),
            'ls': FCMT(input_dim=patch_dim)
        })
        
        # Remove sigmoid from classifier (using BCEWithLogitsLoss)
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # No sigmoid here
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_outputs = {}
        features = []
        
        # Process each scale
        for scale, transform in self.transforms.items():
            x_scale = transform(x)
            patches = self._create_patches(x_scale)
            f_scale, maps = self.fcmts[scale](patches)
            
            batch_outputs[f'maps_{scale}'] = maps
            features.append(f_scale.mean(1))
        
        # Feature fusion and classification (no sigmoid)
        f_combined = torch.cat(features, dim=1)
        output = self.classifier(f_combined)
        
        return output, batch_outputs

    def _create_patches(self, x: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size//2)
        return patches.transpose(1, 2).contiguous().view(B, -1, patch_size * patch_size * C)

class DeepfakeDetectionDataset(Dataset):
    def __init__(self, metadata_path: str, transform: Optional[transforms.Compose] = None, 
                 split: str = 'train'):
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load and validate metadata
        self.samples = self._load_metadata(metadata_path, split)
        self._validate_and_print_stats(split)

    def _load_metadata(self, metadata_path: str, split: str) -> List[Dict]:
        print(f"\nLoading metadata from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if not metadata:
            raise ValueError(f"Metadata file is empty: {metadata_path}")
        
        # Process metadata into samples
        samples = []
        for video_id, frames in metadata.items():
            if not frames:
                continue
                
            is_fake = 'FakeVideo' in video_id
            for frame_data in frames:
                if not isinstance(frame_data, dict) or 'face_path' not in frame_data:
                    continue
                    
                if os.path.exists(frame_data['face_path']):
                    samples.append({
                        'path': frame_data['face_path'],
                        'label': 1 if is_fake else 0,
                        'video_id': video_id,
                        'method': frame_data.get('method', 'real'),
                        'frame_number': frame_data.get('frame_number', 0)
                    })
        
        # Split data
        random.seed(42)
        indices = list(range(len(samples)))
        random.shuffle(indices)
        
        split_map = {
            'train': (0, 0.8),
            'val': (0.8, 0.9),
            'test': (0.9, 1.0)
        }
        start, end = split_map[split]
        split_indices = indices[int(start * len(indices)):int(end * len(indices))]
        
        return [samples[i] for i in split_indices]

    def _validate_and_print_stats(self, split: str) -> None:
        if not self.samples:
            raise ValueError(f"No valid samples found for {split} split!")
            
        n_fakes = sum(1 for s in self.samples if s['label'] == 1)
        n_reals = len(self.samples) - n_fakes
        
        print(f"\nLoaded {len(self.samples)} samples for {split} split")
        print(f"Class distribution: {n_fakes} fake, {n_reals} real")
        
        methods = {}
        for s in self.samples:
            if s['label'] == 1:
                methods[s['method']] = methods.get(s['method'], 0) + 1
        
        print("\nFake methods distribution:")
        for method, count in methods.items():
            print(f"{method}: {count} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['path']}: {str(e)}")
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(sample['label'], dtype=torch.float32)

def train_model(model: nn.Module, train_loader: DataLoader, 
                val_loader: DataLoader, num_epochs: int = 10,
                device: torch.device = None) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create save directory
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=num_epochs, steps_per_epoch=len(train_loader)
    )
    
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Track best model
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (data, labels) in enumerate(pbar):
            try:
                data = data.to(device)
                labels = labels.to(device).view(-1, 1)
                
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs, _ = model(data)
                    loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except RuntimeError as e:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                continue
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device).view(-1, 1)
                outputs, _ = model(data)
                val_loss += criterion(outputs, labels).item()
                
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_accuracy = ((val_preds > 0.5) == val_labels).mean()
        
        # Print metrics
        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        
        # Save checkpoint after each epoch
        checkpoint_path = save_dir / f"model_epoch_{epoch+1}.pt"
        save_model(model, str(checkpoint_path))
        print(f"Saved model checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_path = save_dir / "best_model.pt"
            save_model(model, str(best_model_path))
            print(f"New best model saved with validation accuracy: {val_accuracy:.4f}")

def save_model(model: nn.Module, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'fcmts_state': {k: v.state_dict() for k, v in model.fcmts.items()}
    }, save_path)

def load_model(model: nn.Module, load_path: str) -> nn.Module:
    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    for k, v in checkpoint['fcmts_state'].items():
        model.fcmts[k].load_state_dict(v)
    return model