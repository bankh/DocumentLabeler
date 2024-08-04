#!/bin/bash

# Set the directory paths for the source and target files
src_dir="/mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/boxes_and_transcripts"
tgt_dir="/mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/images"

# Initialize a counter for the number of missing target files
missing_tgt_counter=0

# Loop through each source file and compare with corresponding target file
for src_file in ${src_dir}/*.tsv; do
  # Extract the filename without extension
  file_name=$(basename "${src_file%.*}")
  tgt_file="${tgt_dir}/${file_name}.jpg"
  
  # Check if the target file exists
  if [ -f "${tgt_file}" ]; then
    # Compare the files using diff command and store output in a variable
    diff_output=$(diff "${src_file}" "${tgt_file}")
    
    # Check if diff command output is empty, meaning files are identical
    if [ -z "${diff_output}" ]; then
      echo "Files ${src_file} and ${tgt_file} are identical."
    else
      echo "Files ${src_file} and ${tgt_file} are different."
    fi
  else
    echo "Target file ${tgt_file} does not exist."
    (( missing_tgt_counter++ ))
  fi
done

echo "Total missing target files: ${missing_tgt_counter}"