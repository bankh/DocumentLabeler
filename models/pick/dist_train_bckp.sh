#!/bin/bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=7 \
       --master_addr=localhost --master_port=5556 \
       train.py -c /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/config.json\
#		--resume /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/models/PICK_Default/test_0421_155912/model_best.pth\
		-d 1,2,3,4,5,6,7 --local_world_size 7
