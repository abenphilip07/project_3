import torch
import matplotlib.pyplot as plt
from pathlib import Path
from msfr import MultiScaleSpatialRepresentation
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def visualize_patches(image_tensor, patch_size=32, stride=16):
    """Visualize how an image is divided into patches"""
    # Convert tensor to numpy for visualization
    image = image_tensor.permute(1, 2, 0).numpy()
    h, w = image.shape[:2]
    
    # Create patches visualization
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(image)
    
    # Draw patch grid
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            rect = plt.Rectangle((j, i), patch_size, patch_size, 
                               fill=False, color='red', linewidth=1)
            ax.add_patch(rect)
    
    ax.set_title(f'Image with {patch_size}x{patch_size} patches (stride={stride})')
    return fig

def visualize_stride_example(image_tensor, patch_size=32, stride=16):
    """
    Visualize how different stride values affect patch overlap
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Stride = 32 (no overlap)
    ax1.imshow(image_tensor.permute(1, 2, 0))
    for i in range(0, 224, 32):
        for j in range(0, 224, 32):
            rect = plt.Rectangle((j, i), 32, 32, fill=False, color='red')
            ax1.add_patch(rect)
    ax1.set_title('Stride = 32 (No Overlap)')
    
    # Stride = 16 (50% overlap)
    ax2.imshow(image_tensor.permute(1, 2, 0))
    for i in range(0, 224, 16):
        for j in range(0, 224, 16):
            rect = plt.Rectangle((j, i), 32, 32, fill=False, color='red')
            ax2.add_patch(rect)
    ax2.set_title('Stride = 16 (50% Overlap)')

def visualize_scales(image_path):
    """Visualize different scales of the image and their patches"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image)
    
    # Create MSFR model to use its transforms
    model = MultiScaleSpatialRepresentation(input_size=(224, 224))
    
    # Get different scales
    scales = {
        'Small Scale (112x112)': model.transforms['ss'](img_tensor),
        'Medium Scale (224x224)': model.transforms['ms'](img_tensor),
        'Large Scale (448x448)': model.transforms['ls'](img_tensor)
    }
    
    # Visualize all scales and their patches
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Multi-Scale Spatial Representation Analysis', fontsize=16)
    
    for idx, (scale_name, scaled_img) in enumerate(scales.items()):
        # Original scaled image
        ax = axes[0, idx]
        ax.imshow(scaled_img.permute(1, 2, 0))
        ax.set_title(f'{scale_name}\nShape: {scaled_img.shape[1:]}')
        ax.axis('off')
        
        # Patches visualization
        ax = axes[1, idx]
        ax.imshow(scaled_img.permute(1, 2, 0))
        
        # Draw patch grid
        h, w = scaled_img.shape[1:]
        patch_size = 32
        stride = patch_size // 2
        
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                rect = plt.Rectangle((j, i), patch_size, patch_size, 
                                   fill=False, color='red', linewidth=1)
                ax.add_patch(rect)
        
        ax.set_title(f'Patches ({patch_size}x{patch_size}, stride={stride})')
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Set paths
    project_root = Path(__file__).parent.parent
    image_path = project_root / "data/processed_faces/id00076_FakeVideo-FakeAudio_African_men_frame0_face0.jpg"
    output_dir = project_root / "results/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Create visualizations
    fig = visualize_scales(image_path)
    
    # Save figure
    output_path = output_dir / "msfr_scales_visualization.png"
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    main()