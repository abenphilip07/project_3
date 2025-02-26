import json
import cv2
import dlib
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt

def draw_landmarks(image, shape):
    """Draw the 68 facial landmarks on the image"""
    # Convert image to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw all 68 landmark points
    for i in range(68):
        x, y = shape.part(i).x, shape.part(i).y
        cv2.circle(image_rgb, (x, y), 2, (0, 255, 0), -1)
        # Add point number for reference
        cv2.putText(image_rgb, str(i), (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    return image_rgb

def visualize_landmarks(face_path, output_path, predictor):
    """Load image, detect face, draw landmarks, and save result"""
    # Read image
    img = cv2.imread(str(face_path))
    if img is None:
        print(f"Could not read image: {face_path}")
        return False
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    
    if len(faces) == 0:
        print(f"No face detected in: {face_path}")
        return False
    
    # Get landmarks for first face
    shape = predictor(gray, faces[0])
    
    # Draw landmarks
    result = draw_landmarks(img.copy(), shape)
    
    # Save result
    plt.figure(figsize=(8, 8))
    plt.imshow(result)
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()
    
    return True

def main():
    # Setup paths
    metadata_path = Path("E:/project_3/metadata/processed_faces_metadata.json")
    predictor_path = Path("E:/project_3/models/shape_predictor_68_face_landmarks.dat")
    output_dir = Path("E:/project_3/results/dlib")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load face predictor
    predictor = dlib.shape_predictor(str(predictor_path))
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get all face paths
    all_faces = []
    for video_data in metadata.values():
        for face in video_data:
            if 'face_path' in face:
                all_faces.append(face['face_path'])
    
    # Select 20 random faces
    selected_faces = random.sample(all_faces, min(20, len(all_faces)))
    
    # Process each selected face
    for i, face_path in enumerate(selected_faces):
        output_path = output_dir / f"landmark_face_{i}.png"
        print(f"Processing face {i+1}/20: {face_path}")
        
        if visualize_landmarks(Path(face_path), output_path, predictor):
            print(f"Saved landmark visualization to: {output_path}")
        else:
            print(f"Failed to process: {face_path}")

if __name__ == "__main__":
    main()