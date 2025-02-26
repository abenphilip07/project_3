import sys
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import required modules from project
from modules.msfr import MultiScaleSpatialRepresentation, load_model
from modules.train import DeepfakeDetectionDataset, select_random_batch


# Add after imports
project_root = Path(__file__).parent.parent

def main():
    # Setup device and paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = project_root / "saved_models/best_model.pt"
    metadata_path = project_root / "metadata/processed_faces_metadata_cleaned.json"
    output_dir = project_root / "results/predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and load model
    model = MultiScaleSpatialRepresentation(input_size=(224, 224))
    model = load_model(model, str(model_path))
    model = model.to(device)
    model.eval()

    # Create dataset and dataloader
    dataset = DeepfakeDetectionDataset(metadata_path=str(metadata_path), split='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Get 10 random images
    images, labels = select_random_batch(dataloader, num_images=10)
    
    # Process images and get predictions
    with torch.no_grad():
        images = images.to(device)
        outputs, _ = model(images)
        predictions = torch.sigmoid(outputs)

    # Visualize results
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()

    for idx in range(10):
        img = images[idx].cpu()
        img = img.permute(1, 2, 0)  # Change from CxHxW to HxWxC
        pred = predictions[idx].item()
        true_label = labels[idx].item()
        
        axes[idx].imshow(img)
        axes[idx].set_title(f'Pred: {"Fake" if pred > 0.5 else "Real"}\nTrue: {"Fake" if true_label > 0.5 else "Real"}\nConf: {pred:.2f}')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'random_predictions.png')
    plt.close()

    print(f"Results saved to {output_dir / 'random_predictions.png'}")

if __name__ == "__main__":
    main()