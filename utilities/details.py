import json
from pathlib import Path
from collections import Counter

def check_video_details(json_path):
    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize counters
    total_videos = len(data)
    missing_videos = []
    type_counter = Counter()
    
    # Check each video and count types
    for item in data:
        video_path = Path(item['path'])
        type_counter[item['type']] += 1
        
        if not video_path.exists():
            missing_videos.append(str(video_path))
    
    # Print results
    print("\n=== Video Statistics ===")
    print(f"Total videos in metadata: {total_videos}")
    
    print("\n=== Videos by Type ===")
    for video_type, count in type_counter.items():
        print(f"{video_type}: {count} videos")
    
    print("\n=== Missing Videos Check ===")
    if missing_videos:
        print(f"Found {len(missing_videos)} missing videos:")
        for path in missing_videos:
            print(f"- {path}")
    else:
        print("All videos exist in the specified paths")

def main():
    json_path = Path("E:/project_3/metadata/metadata_formatted.json")
    
    try:
        check_video_details(json_path)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()