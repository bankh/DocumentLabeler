import os
import sys
import chardet

def process_file(file_path, output_dir):
    # Detect encoding
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    # Read file with detected encoding and ignore errors
    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
        content = file.read()

    # Define output file path
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_dir, file_name)

    # Save the cleaned content to a new file with UTF-8 encoding
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(content)

    print(f"Processed and saved: {output_file_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python clean_text_files.py {target_folder}")
        return

    # Get the target folder from command-line arguments
    target_folder = sys.argv[1]
    
    # Validate the target folder
    if not os.path.isdir(target_folder):
        print("Invalid folder path. Please try again.")
        return

    # Create an output directory
    output_dir = os.path.join(target_folder, 'cleaned_files')
    os.makedirs(output_dir, exist_ok=True)

    # Process each text file in the directory
    for root, _, files in os.walk(target_folder):
        for file in files:
            if file.endswith('.tsv'):
                file_path = os.path.join(root, file)
                process_file(file_path, output_dir)

if __name__ == "__main__":
    main()
