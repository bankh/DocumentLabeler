import json
import argparse

def calculate_total_characters(json_file_path):
    try:
        with open(json_file_path, 'r') as f:
            data = f.read()

        # Extract the JSON part (starts after the first tab)
        json_data = data.split('\t', 1)[1]
        annotations = json.loads(json_data)

        total_characters = sum(len(annotation['transcription']) for annotation in annotations)
        print(f"Total number of characters in transcriptions: {total_characters}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate total number of characters in transcription fields of a JSON file.')
    parser.add_argument('--target_json', type=str, required=True, help='Path to the target JSON file.')
    
    args = parser.parse_args()
    calculate_total_characters(args.target_json)
