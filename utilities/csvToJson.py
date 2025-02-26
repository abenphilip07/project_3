import csv
import json
from pathlib import Path

def csv_to_json(csv_file_path, json_file_path):
    # List to store all records
    data = []
    
    # Read CSV file
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Convert each row to dictionary and append to data list
        for row in csv_reader:
            # Clean up empty target2 values
            if row['target2'] == '-':
                row['target2'] = None
                
            # Convert path to use forward slashes for consistency
            if 'full_path' in row:
                row['full_path'] = row['full_path'].replace('\\', '/')
                
            data.append(row)
    
    # Write JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

def format_json(input_json_path, output_json_path):
    # Read the original JSON file
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create new formatted list
    formatted_data = []
    
    for item in data:
        # Format full path correctly
        full_path = item['full_path'].replace('\\', '/')
        path = item['path']
        complete_path = f"{full_path}/{path}"
        
        # Create new entry with only required fields
        new_entry = {
            'path': complete_path,
            'type': item['type'],
            'method': item['method']
        }
        
        formatted_data.append(new_entry)
    
    # Write the formatted JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)

def main():
    # Define input and output file paths
    csv_file = Path('E:/project_3/metadata/metadata_updated.csv')
    json_file = csv_file.parent / 'metadata_updated.json'
    
    try:
        csv_to_json(csv_file, json_file)
        print(f"Successfully converted CSV to JSON. Output saved to {json_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    input_json = Path('E:/project_3/metadata/metadata_updated.json')
    output_json = input_json.parent / 'metadata_formatted.json'
    
    try:
        format_json(input_json, output_json)
        print(f"Successfully formatted JSON. Output saved to {output_json}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()