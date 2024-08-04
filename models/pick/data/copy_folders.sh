#!/bin/bash

# Usage: ./script.sh <source> <destination> <number_of_tsv_files>

src="$1"
dst="$2"
num_tsv_files="$3"

# Check if the destination folder exists, if not, create it
mkdir -p "$dst"

# Function to copy files with the same basename
copy_same_basename() {
  base="$1"
  src_dir="$2"
  dst_dir="$3"

  find "$src_dir" -type f -name "${base}.*" -exec cp {} "$dst_dir" \;
}

# Count the total number of tsv files in the source folder
total_tsv_files=$(find "$src" -type f -name "*.tsv" | wc -l)
counter=0

# Iterate through the tsv files and copy the required number of files
while IFS= read -r tsv_file; do
  if [ $counter -ge $num_tsv_files ]; then
    break
  fi

  src_subfolder=$(dirname "$tsv_file")
  dst_subfolder="$dst${src_subfolder#$src}"

  # Create the destination subfolder if it doesn't exist
  mkdir -p "$dst_subfolder"

  # Copy the tsv file
  cp "$tsv_file" "$dst_subfolder"

  # Copy the files with the same basename
  base=$(basename "$tsv_file" .tsv)
  for other_subfolder in $(find "$src" -mindepth 1 -type d); do
    if [ "$other_subfolder" != "$src_subfolder" ]; then
      dst_other_subfolder="$dst${other_subfolder#$src}"
      mkdir -p "$dst_other_subfolder"
      copy_same_basename "$base" "$other_subfolder" "$dst_other_subfolder"
    fi
  done

  # Calculate and display the progress
  counter=$((counter+1))
  progress=$((counter*100/num_tsv_files))
  echo "Progress: $progress%"
  echo "Number of processed files: $counter"
  echo "Copied: $tsv_file to $dst_subfolder"

done < <(find "$src" -type f -name "*.tsv")

