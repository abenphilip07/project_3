import sys
from pathlib import Path


# Update imports to use relative imports
from msfr import MultiScaleSpatialRepresentation, load_model, DeepfakeDetectionDataset  
from train import select_random_batch  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np

# Set project root
project_root = Path(__file__).parent.parent

def visualize_attention(img, attention_map, ax, title):
    """
    Display an attention map over an image.
    """
    # Convert tensor image to numpy
    img_np = img.permute(1, 2, 0).cpu().numpy()  
    img_np = np.clip(img_np, 0, 1)  # Ensure values are in valid range
    
    # Process attention map
    if attention_map.dim() == 3:  
        attention_map = attention_map.mean(0)  # Average over heads
    
    attention_map = attention_map.cpu().detach().numpy()  

    # Resize attention map to match image size
    attention_map_resized = TF.resize(torch.tensor(attention_map), img_np.shape[:2], interpolation=TF.InterpolationMode.BILINEAR)
    attention_map_resized = attention_map_resized.numpy()

    # Normalize attention map
    attention_map_resized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min() + 1e-8)

    # Display image
    ax.imshow(img_np)
    ax.imshow(attention_map_resized, cmap='jet', alpha=0.5)  # Overlay attention map
    ax.set_title(title)
    ax.axis("off")


def main():
    """
    Main function to load the model, process images, and visualize attention.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths
    model_path = project_root / "saved_models/best_model.pt"
    metadata_path = project_root / "metadata/processed_faces_metadata_cleaned.json"
    output_dir = project_root / "results/predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model from: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model
    model = MultiScaleSpatialRepresentation(input_size=(224, 224))
    model = load_model(model, str(model_path))
    model = model.to(device)
    model.eval()

    # Load dataset
    dataset = DeepfakeDetectionDataset(metadata_path=str(metadata_path), split='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Select 10 random images
    images, labels = select_random_batch(dataloader, num_images=10)

    with torch.no_grad():
        images = images.to(device)
        outputs, _ = model(images)

        # Apply sigmoid activation with temperature scaling
        predictions = torch.sigmoid(outputs)

    # Set up visualization grid
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for idx in range(10):
        img = images[idx].cpu()
        
        # Ensure the image format is correct
        img = img.permute(1, 2, 0)  # Convert (C, H, W) → (H, W, C)
        img = torch.clamp(img, 0, 1)

        pred = predictions[idx].item()
        true_label = labels[idx].item()

        # Titles for predictions
        pred_text = f'Pred: {"Fake" if pred > 0.5 else "Real"} ({pred:.2f})'
        true_text = f'True: {"Fake" if true_label > 0.5 else "Real"}'
        color = 'red' if pred > 0.5 != true_label > 0.5 else 'green'

        # Plot Original Image
        axes[idx].imshow(img)
        axes[idx].set_title(f'Original Image\n{pred_text}\n{true_text}', color=color)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'predictions.png')
    plt.close()

if __name__ == "__main__":
    main()
