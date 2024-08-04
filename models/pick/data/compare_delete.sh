#!/bin/bash

# Set the directory paths for the source and target files
tgt_dir="/mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/boxes_and_transcripts"
src_dir="/mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/images"

# Get a list of all files in the target directory
tgt_file_list=$(ls "${tgt_dir}")

# Loop through each file in the target directory
for tgt_file in ${tgt_file_list}; do
  # Get the base filename of the target file
  tgt_base_file=$(basename "${tgt_file}")
  tgt_base_file="${tgt_base_file%.*}" # Remove the extension

  # Check if there is a matching source file
  src_file="${src_dir}/${tgt_base_file}.tsv"
  if [ -f "${src_file}" ]; then
    # There is a matching source file, so skip the deletion of the target file
    continue
  fi

  # If we didn't find a matching source file, delete the target file
  echo "Deleting ${tgt_file} because there is no matching source file."
  rm "${tgt_dir}/${tgt_file}"
done
