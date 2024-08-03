#!/bin/bash
#sudo docker exec -it 39412c41b6b5 source activate documentlabeler | python /mnt/data_drive/CSU_PhD/research/software/DocumentLabeler/DocumentLabel/DocumentLabeler.py --kie True
docker exec -it 17918353f327 bash -c 'source activate py_3.8 && python /mnt/data_drive/CSU_PhD/research/software/DocumentLabeler/DocumentLabel/DocumentLabeler.py --kie True'

