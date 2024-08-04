import os
import shutil
from tqdm import tqdm

# Define the root folder where your TSV files are located
root_folder = '/mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000'

# Define the output folder where the new TSV files will be saved
output_folder = '/mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy'

# Walk through the root folder and its subfolders
for dirpath, dirnames, filenames in os.walk(root_folder):
    # Create the corresponding subdirectory in the output folder
    relative_dirpath = os.path.relpath(dirpath, root_folder)
    new_dirpath = os.path.join(output_folder, relative_dirpath)
    if not os.path.exists(new_dirpath):
        os.makedirs(new_dirpath)
    # Loop through the filenames in the current folder
    for filename in tqdm(filenames):
        # Check if the file has a .tsv extension
        if filename.endswith('.tsv'):
            # Construct the full path to the file
            filepath = os.path.join(dirpath, filename)
            # Read the contents of the file
            with open(filepath, 'r') as f:
                lines = f.readlines()
            # Remove the last column from each line
            lines = [line.strip().split(',')[:-1] for line in lines]
            # Write the new contents to a new file with '_new' appended to the filename
            new_filename = filename[:-4] + '.tsv'
            new_filepath = os.path.join(new_dirpath, new_filename)
            with open(new_filepath, 'w') as f:
                f.write('\n'.join([','.join(line) for line in lines]))
