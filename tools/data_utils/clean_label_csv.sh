#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: bash script.sh <source_directory> <target_directory>"
  exit 1
fi

source_directory="$1"
target_directory="$2"

# Check if source directory exists
if [ ! -d "$source_directory" ]; then
  echo "Source directory does not exist: $source_directory"
  exit 1
fi

# Create target directory if it doesn't exist
if [ ! -d "$target_directory" ]; then
  echo "Target directory does not exist. Creating directory: $target_directory"
  mkdir -p "$target_directory"
fi

# Iterate over each .tsv file in the source directory
for file in "$source_directory"/*.tsv; do
    filename=$(basename "$file")
    target_file="$target_directory/$filename"

    # Remove the last component from each line
    sed 's/,[^,]*$//' "$file" > "$target_file"

    echo "Modified file created: $target_file"
done
