#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

def merge_json_files():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Input files
    input_files = [
        os.path.join(current_dir, 'tool_descriptions.json'),
        os.path.join(current_dir, 'tool_descriptions2.json'),
        os.path.join(current_dir, 'tool_descriptions3.json')
    ]
    
    # Output file path
    output_file = os.path.join(parent_dir, 'tool_descriptions.json')
    
    # Merged data
    merged_data = {}
    
    # Read and merge each JSON file
    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Successfully loaded {file_path}")
                
                # If merged_data is empty, initialize it with the first file's structure
                if not merged_data:
                    merged_data = data
                else:
                    # Merge based on the structure of the JSON files
                    # This assumes the JSON files have similar structure
                    # Adjust this logic if the structure is different
                    if isinstance(data, dict):
                        merged_data.update(data)
                    elif isinstance(data, list):
                        if isinstance(merged_data, list):
                            merged_data.extend(data)
                        else:
                            merged_data = data
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Write the merged data to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully merged files to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    merge_json_files() 