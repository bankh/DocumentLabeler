#!/bin/bash
# python test.py --checkpoint /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/models/PICK_Default/test_0424_043653/checkpoint-epoch60.pth\
#                --boxes_transcripts /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/boxes_and_transcripts/ \
#                --images_path /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/images/ \
#                --output_folder /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test_output_bs32/ \
#                --gpu -1 \
#                --batch_size 24

python test.py --checkpoint /mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/models/pick/models/PICK_Default/SROIE_test_0923_025525/checkpoint-epoch120.pth \
               --bt /mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/models/pick/data/McMaster/McMasterCarr_DL_200-500_pp_PICK/boxes_and_transcripts/ \
               --impt /mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/models/pick/data/McMaster/McMasterCarr_DL_200-500_pp_PICK/images/ \
               --output /mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/models/pick/data/McMaster/McMasterCarr_DL_200-500_pp_PICK/ \
               --gpu -1 \
               --bs 12