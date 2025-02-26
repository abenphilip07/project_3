import os
import json
import cv2
import dlib
import numpy as np
import bz2
import shutil
from pathlib import Path

# Update predictor paths
compressed_path = r"E:\Project_3\models\shape_predictor_68_face_landmarks.dat.bz2"
predictor_path = r"E:\Project_3\models\shape_predictor_68_face_landmarks.dat"

# Extract the compressed file 
if not os.path.exists(predictor_path):
    print("Extracting shape predictor file...")
    with bz2.BZ2File(compressed_path) as fr, open(predictor_path, 'wb') as fw:
        shutil.copyfileobj(fr, fw)
    print("Extraction complete.")

# Load dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)  

# Define input metadata path and output directories
metadata_path = r"E:\project_3\metadata\metadata_formatted.json"
output_faces_dir = r"E:\Project_3\data\processed_faces"
os.makedirs(output_faces_dir, exist_ok=True)

def align_face(image, face_rect):
    """Align face using dlib facial landmarks."""
    shape = predictor(image, face_rect)
    points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

    # Compute the center of eyes for alignment
    left_eye = points[36:42].mean(axis=0)
    right_eye = points[42:48].mean(axis=0)

    # Compute rotation angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Compute center and scale
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    scale = 1.2

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return aligned_face

def process_video(video_info):
    """Process a single video and extract faces."""
    video_metadata = []
    video_path = video_info['path']
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for i, face in enumerate(faces):
            aligned_face = align_face(frame, face)
            face_filename = f"{video_id}_frame{frame_count}_face{i}.jpg"
            face_path = os.path.join(output_faces_dir, face_filename)
            
            cv2.imwrite(face_path, aligned_face)
            
            # Store metadata
            video_metadata.append({
                "frame_number": frame_count,
                "face_index": i,
                "face_path": face_path,
                "video_type": video_info['type'],
                "method": video_info['method']
            })
        
        frame_count += 1
    
    cap.release()
    return video_metadata

def main():
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Process all videos
    all_metadata = {}
    
    for video_info in metadata:
        video_id = os.path.splitext(os.path.basename(video_info['path']))[0]
        print(f"Processing video: {video_id}")
        
        try:
            video_metadata = process_video(video_info)
            all_metadata[video_id] = video_metadata
        except Exception as e:
            print(f"Error processing video {video_id}: {str(e)}")
    
    # Save metadata
    metadata_output = os.path.join(output_faces_dir, "processed_faces_metadata.json")
    with open(metadata_output, "w") as f:
        json.dump(all_metadata, f, indent=4)
    
    print(f"Processing complete. Faces saved in {output_faces_dir}")

if __name__ == "__main__":
    main()
