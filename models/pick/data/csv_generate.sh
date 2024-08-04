#!/bin/bash

# Usage: ./script.sh <source_csv> <target_csv> <boxes_and_transcripts_folder>

source_csv="$1"
target_csv="$2"
boxes_and_transcripts_folder="$3"

# Initialize the target CSV file
touch "$target_csv"

# Find TSV files in the boxes_and_transcripts folder
tsv_files=$(find "$boxes_and_transcripts_folder" -type f -name "*.tsv")

# Initialize the row counter
row_counter=1

# Process each TSV file
for tsv_file in $tsv_files; do
  # Extract the filename without extension
  base=$(basename "$tsv_file" .tsv)

  # Find the corresponding line in the source CSV file
  matching_line=$(grep -F -w "$base" "$source_csv")
  if [ -n "$matching_line" ]; then
    # Replace the first number in the first item with the row counter
    updated_line=$(echo "$matching_line" | sed "s/^[0-9]\+/$row_counter/")

    # Append the updated line to the target CSV file
    echo "$updated_line" >> "$target_csv"

    # Increment the row counter
    row_counter=$((row_counter+1))
  fi
done
