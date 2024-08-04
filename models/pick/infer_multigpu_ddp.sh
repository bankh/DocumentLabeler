#!/bin/bash
# export NCCL_SOCKET_IFNAME=^docker0,lo
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

python -m torch.distributed.launch --nproc_per_node=6 --nnodes=1 --node_rank=0 test_multigpu_ddp.py \
                --checkpoint /mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/models/pick/models/PICK_Default/SROIE_test_0925_081333/model_best.pth \
                --bt /mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/models/pick/data/McMaster/McMasterCarr_DL_200-500_pp_PICK/boxes_and_transcripts/ \
                --impt /mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/models/pick/data/McMaster/McMasterCarr_DL_200-500_pp_PICK/images/ \
                --output /mnt/data_drive/CSU_PhD/research/software/DocumentEngineering/DocumentLabeler_FinalDev/models/pick/data/McMaster/McMasterCarr_DL_200-500_pp_PICK/ \
                --bs 2

            #    --checkpoint /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/models/PICK_Default/test_0424_043653/checkpoint-epoch60.pth \             
            #    --bt /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/boxes_and_transcripts/ \
            #    --impt /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test/images/ \
            #    --output /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000_copy/test_output_bs32/ \
               
