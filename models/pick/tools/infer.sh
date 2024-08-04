#!/bin/bash
python test.py --checkpoint /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/models/PICK_Default/test_0424_043653/checkpoint-epoch60.pth\
               --boxes_transcripts /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/boxes_and_transcripts/ \
               --images_path /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/images/ \
               --output_folder /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test_output_bs32/ \
               --gpu -1 \
               --batch_size 24
