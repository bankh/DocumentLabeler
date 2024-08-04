#!/bin/bash

# Usage: ./evaluate_labels.sh <target_dir> <label_list>

target_dir="$1"
label_list="$2"

# Read label list and initialize counts
declare -A label_counts
while IFS= read -r label; do
    label_counts["$label"]=0
done < "$label_list"

# Iterate through TSV files and count labels
for filename in "$target_dir"/*.tsv; do
    while IFS= read -r line; do
        label="${line##*,}"
        label_counts["$label"]=$(( label_counts["$label"] + 1 ))
    done < "$filename"
done

# Calculate total number of labels
total_count=0
for count in "${label_counts[@]}"; do
    total_count=$((total_count + count))
done

# Print label counts as a table
printf "%-20s | %-10s | %-10s\n" "Label" "Count" "Percentage"
printf "%s\n" "---------------------+------------+------------"
for label in "${!label_counts[@]}"; do
    count="${label_counts["$label"]}"
    percentage=$(printf "%.2f" "$(echo "scale=2; $count*100/$total_count" | bc)")
    printf "%-20s | %-10s | %-10s%%\n" "$label" "$count" "$percentage"
done
printf "%-20s | %-10s | %-10s%%\n" "Total" "$total_count" "100"
