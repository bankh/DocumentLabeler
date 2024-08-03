#!/bin/bash
python -m torch torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=7 \
       --master_addr=localhost --master_port=5556 \
       train.py -c ./config.json \
       -d 1,2,3,4,5,6,7 --local_world_size 7
