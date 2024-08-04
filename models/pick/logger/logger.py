# -*- coding: utf-8 -*-

import logging
import logging.config
from pathlib import Path
from utils import read_json
import torch


def setup_logging(save_dir, log_config='logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(__file__).parent.joinpath(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)


def log_gpu_memory(stage, step_idx):
        for device_id in range(torch.cuda.device_count()):
            print(f"[Step {step_idx}] Memory allocated on GPU {device_id} during {stage}: {torch.cuda.memory_allocated(device_id)/1e9} GBs")

