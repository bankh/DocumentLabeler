import os
import sys
import matplotlib.pyplot as plt

def extract_categories_from_tsv(file_path):
    """Extract categories from a TSV file."""
    categories = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split by tab and then take the last element as category
            columns = line.strip().split('\t')
            category = columns[-1].split(',')[-1]
            categories.append(category)
    return categories

def main(directory_path):
    all_categories = []

    # Iterate over all files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.tsv'):
            file_path = os.path.join(directory_path, file_name)
            categories = extract_categories_from_tsv(file_path)
            all_categories.extend(categories)

    # Count the occurrences of each category
    category_counts = {}
    for category in all_categories:
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1

    
    # Sort the categories by their counts in descending order
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_keys = [item[0] for item in sorted_categories]
    sorted_values = [item[1] for item in sorted_categories]


    # Plot the categories
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_keys, 
                   sorted_values, 
                   color=['red', 'green', 'blue', 'yellow', 'purple', 'cyan'])
    plt.xticks(rotation=90)
    plt.ylabel('Number of Occurrences')
    plt.title('Category Counts')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    total = sum(sorted_values)

    for bar in bars:
        height = bar.get_height()
        percentage = (height / total) * 100

        # Determine position, color, and vertical alignment based on percentage
        if percentage < 20:
            y_position = height + 10
            text_color = 'black'
            vertical_alignment = 'bottom'
        else:
            y_position = height / 2
            text_color = 'white'
            vertical_alignment = 'center'

        plt.text(bar.get_x() + bar.get_width() / 2, 
                 y_position, 
                 f'{height}\n  ({percentage:.2f}%)', 
                 ha='center', 
                 va=vertical_alignment,
                 rotation=90,
                 color=text_color)

    plt.show()

    # List unique categories
    unique_categories = set(all_categories)
    print("Unique Categories:")
    for category in unique_categories:
        print(category)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <directory_path>")
        sys.exit(1)
    directory = sys.argv[1]
    main(directory)
