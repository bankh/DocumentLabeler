#!/bin/bash

# Set the directory path for the files
source_dir="/mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/boxes_and_transcripts"

cd "$source_dir"

# Loop through each file and rename it
for file_name in *_new.*; do
    # Remove "_new" from the filename
    new_file_name=$(echo "$file_name" | sed 's/_new//')

    # Rename the file
    mv "$file_name" "$new_file_name"
done
