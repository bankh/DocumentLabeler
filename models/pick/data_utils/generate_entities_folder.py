import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def process_file(input_file, output_folder, keys):
    # Create the output file path
    output_file = os.path.join(output_folder, os.path.basename(input_file).replace('.tsv', '.txt'))

    # Open the input file and create the output file
    with open(input_file, 'r') as tsvfile, open(output_file, 'w') as outfile:
        # Create a CSV reader for the input file
        reader = csv.reader(tsvfile, delimiter=',')
        # Extract the keys from the last word of each row
        header = next(reader)
        keys = [row[-1] for row in reader]
        tsvfile.seek(0)  # Reset the file pointer
        # Create an empty dictionary to store the extracted information
        data = {key: [] for key in keys}

        # Loop through each line in the input file
        for row in reader:
            # Extract the relevant fields from the row
            text = row[9]
            category = row[10]
            # Use the last item of the row as the key
            key = row[-1]
            # Process the text based on the category
            if category == key:
                if key == keys[2]:
                    # For the 'address' key, append the text to the existing list
                    data[key].append(text)
                elif key == keys[1]:
                    # For the 'date' key, extract only the date part from the text
                    data[key] = [text.split(' ')[0]]
                else:
                    # For all other keys, append the text to the existing list
                    data[key].append(text)

        # Convert the list values to string format
        for key, value in data.items():
            data[key] = ' '.join(value)

        # Convert the dictionary to JSON format
        json_data = json.dumps(data)
        # Write the JSON data to the output file
        outfile.write(json_data)

    return output_file

if __name__ == '__main__':
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Process boxed_and_transcriptions files.')
    parser.add_argument('--input_folder', type=str, help='the input folder for boxed_and_transcriptions')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='the number of worker processes')
    args = parser.parse_args()

    # Find all TSV files in the input folder
    input_files = [os.path.join(args.input_folder, filename) for filename in os.listdir(args.input_folder) if filename.endswith('.tsv')]

    # Extract the keys from the last column of the first TSV file found in the input folder
    with open(input_files[0], 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter=',')
        keys = [row[-1] for row in reader]

    # Create the output folder
    output_folder = os.path.join(os.path.dirname(os.path.dirname(args.input_folder)), 'entities')
    os.makedirs(output_folder, exist_ok=True)

    # Use thread pool to process the files in parallel
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for input_file in input_files:
            future = executor.submit(process_file, input_file, output_folder, keys)
            futures.append((future, input_file))

        # Wait for all the futures to complete and track progress
        results