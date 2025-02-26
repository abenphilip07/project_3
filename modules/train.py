import torch
from pathlib import Path
from msfr import MultiScaleSpatialRepresentation, train_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
import random

class DeepfakeDetectionDataset(Dataset):
    def __init__(self, metadata_path, transform=None, split='train'):
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load metadata
        print(f"\nLoading metadata from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        if not self.metadata:
            raise ValueError(f"Metadata file is empty: {metadata_path}")
            
        # Create samples list with video IDs and frames
        self.samples = []
        for video_id, frames in self.metadata.items():
            if not frames:
                continue
                
            is_fake = 'FakeVideo' in video_id
            
            for frame_data in frames:
                if not isinstance(frame_data, dict) or 'face_path' not in frame_data:
                    continue
                    
                # Use the actual path from metadata
                image_path = frame_data['face_path']
                
                self.samples.append({
                    'path': image_path,
                    'label': 1 if is_fake else 0,
                    'video_id': video_id,
                    'method': frame_data.get('method', 'real'),
                    'frame_number': frame_data.get('frame_number', 0)
                })
        
        # Split data
        random.seed(42)  # for reproducibility
        n_samples = len(self.samples)
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        if split == 'train':
            indices = indices[:int(0.8 * n_samples)]
        elif split == 'val':
            indices = indices[int(0.8 * n_samples):int(0.9 * n_samples)]
        else:  # test
            indices = indices[int(0.9 * n_samples):]
            
        self.samples = [self.samples[i] for i in indices]
        
        # Print statistics
        n_fakes = sum(1 for s in self.samples if s['label'] == 1)
        n_reals = len(self.samples) - n_fakes
        print(f"\nLoaded {len(self.samples)} samples for {split} split")
        print(f"Class distribution: {n_fakes} fake, {n_reals} real")
        
        # Print methods distribution
        methods = {}
        for s in self.samples:
            if s['label'] == 1:
                methods[s['method']] = methods.get(s['method'], 0) + 1
        print("\nFake methods distribution:")
        for method, count in methods.items():
            print(f"{method}: {count} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample['path']).convert('RGB')
        except:
            print(f"Error loading image: {sample['path']}")
            # Return a blank image in case of error
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(sample['label'], dtype=torch.float32)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    metadata_path = "E:/project_3/metadata/processed_faces_metadata_cleaned.json"
    save_dir = "E:/project_3/saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # Create datasets and dataloaders
    batch_size = 32
    train_dataset = DeepfakeDetectionDataset(metadata_path=metadata_path, split='train')
    val_dataset = DeepfakeDetectionDataset(metadata_path=metadata_path, split='val')
    
    # Print class distribution
    train_fakes = sum(1 for sample in train_dataset.samples if sample['label'] == 1)
    train_reals = len(train_dataset) - train_fakes
    print(f"\nTraining set distribution:")
    print(f"Fake videos: {train_fakes}")
    print(f"Real videos: {train_reals}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4)

    # Initialize and train model
    model = MultiScaleSpatialRepresentation(input_size=(224, 224))
    model = model.to(device)
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10
    )

if __name__ == "__main__":
    main()