import os
from tqdm import tqdm

# Set the directory paths for the source and target files
tgt_dir = "/mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/boxes_and_transcripts"
src_dir = "/mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/images"

# Get a list of all files in the source and target directories
src_file_list = os.listdir(src_dir)
tgt_file_list = os.listdir(tgt_dir)

# Check for missing TSV files in the source directory
for tgt_file in tgt_file_list:
    # Check if it's a JPG file
    if not tgt_file.endswith(".jpg"):
        continue

    # Get the base filename of the target file
    tgt_base_file = os.path.splitext(tgt_file)[0]

    # Check if there is a matching TSV file
    src_file = os.path.join(src_dir, f"{tgt_base_file}.tsv")
    if not os.path.isfile(src_file):
        # There is no matching TSV file, so delete the target file
        print(f"Deleting {tgt_file} because there is no matching source file.")
        os.remove(os.path.join(tgt_dir, tgt_file))

# Check for missing JPG files in the target directory
for src_file in src_file_list:
    # Check if it's a TSV file
    if not src_file.endswith(".tsv"):
        continue

    # Get the base filename of the source file
    src_base_file = os.path.splitext(src_file)[0]

    # Check if there is a matching JPG file
    tgt_file = os.path.join(tgt_dir, f"{src_base_file}.jpg")
    if not os.path.isfile(tgt_file):
        # There is no matching JPG file, so delete the source file
        print(f"Deleting {src_file} because there is no matching target file.")
        os.remove(os.path.join(src_dir, src_file))
    else:
        # There is a matching JPG file, so skip the deletion of the source file
        continue
