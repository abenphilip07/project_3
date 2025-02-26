import json
from pathlib import Path

def remove_face_index(json_path):
    """Remove face_index field from metadata"""
    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Clean metadata
    for video_id, frames in data.items():
        for frame in frames:
            if 'face_index' in frame:
                del frame['face_index']
    
    # Save cleaned metadata
    output_path = json_path.parent / 'processed_faces_metadata_cleaned.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Cleaned metadata saved to: {output_path}")

def main():
    json_path = Path("E:/project_3/metadata/processed_faces_metadata.json")
    
    try:
        remove_face_index(json_path)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()