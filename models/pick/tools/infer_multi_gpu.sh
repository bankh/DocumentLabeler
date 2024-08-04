#!/bin/bash

# Specify the number of GPUs to use
NGPUS=7

# Specify the path to the script
SCRIPT_PATH=/mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/test_multi_gpu.py

# Specify the command-line arguments
# CHECKPOINT=/mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/models/PICK_Default/test_0424_043653/checkpoint-epoch60.pth
# CHECKPOINT=/mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/models/pick/models/PICK_Default/SROIE_test_0923_025525/checkpoint-epoch120.pth
CHECKPOINT=/mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/models/pick/models/PICK_Default/SROIE_test_0923_184405/model_best.pth
BT=/mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/boxes_and_transcripts/
IMPT=/mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/images/
OUTPUT_FOLDER=/mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test_output_bs24_multi/
BS=24
WORLD_SIZE=$NGPUS

# Run the script on multiple GPUs
python -m torch.distributed.launch --nproc_per_node=$NGPUS \
                $SCRIPT_PATH --checkpoint $CHECKPOINT \
                             --bt $BT \
                             --impt $IMPT \
                             --output_folder $OUTPUT_FOLDER \
                             --bs $BS \
                             --world_size $WORLD_SIZE