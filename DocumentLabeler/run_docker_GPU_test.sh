#!/bin/bash
#docker exec -it 39412c41b6b5 watch -n 1 rocm-smi # Test rocm-smi
#docker exec -it 39412c41b6b5 bash -c conda activate documentlabeler | pip list | grep torch
#docker exec -it 39412c41b6b5 bash -c conda activate documentlabeler | python -c "import torch; print(torch.cuda.is_available())" # Test python
docker exec -it 39412c41b6b5 bash -c 'source activate documentlabeler && python -c "import torch; print(torch.cuda.is_available())"'
