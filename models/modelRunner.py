# -#- coding: utf-8 -*-
# @Author: Sinan Bank
# @Created Time: 8/9/2024 3:43 PM
import logging
import sys
import os
from pathlib import Path
import yaml
import torch

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from pick.tools.inference.pickInference import PICKInference
from tools.infer.utility import check_gpu

logger = logging.getLogger(__name__)

class Inference:
    def __init__(self, model_selection, debug_mode=False, **kwargs):
        self.model_selection = model_selection
        self.debug_mode = debug_mode  # Store debug_mode
        self.params = kwargs
        
        logger.info(f"self.params.keys(): {self.params.keys()}")

        config_file = self.params.get('config_file')
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as config_data:
                config_data = yaml.safe_load(config_data)
            self.checkpoint = config_data['checkpoint']
            self.bt = config_data['bt']
            self.impt = config_data['impt']
            self.bs = config_data.get('bs', 16)
            self.output = config_data.get('output', 'output')
            self.local_rank = config_data.get('local_rank', 0)
            self.num_workers = config_data.get('num_workers', 2)
            self.use_cpu = config_data.get('use_cpu', False)
        else:
            self.checkpoint = self.params.get('checkpoint_path')
            self.bt = self.params.get('bt_path')
            self.impt = self.params.get('impt_path')
            self.bs = self.params.get('batch_size', 16)
            self.output = self.params.get('output_path', 'output')
            self.local_rank = self.params.get('local_rank', 0)
            self.num_workers = self.params.get('num_workers', 2)
            self.use_cpu = self.params.get('use_cpu', False)
        
        self.multi_gpu = self.params.get('multi_gpu', False)

        self.device = torch.device("cpu" if self.use_cpu else "cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"self.checkpoint: {self.checkpoint}")
        logger.info(f"self.bt: {self.bt}")
        logger.info(f"self.impt: {self.impt}")
        logger.info(f"self.bs: {self.bs}")
        logger.info(f"self.output: {self.output}")
        logger.info(f"self.local_rank: {self.local_rank}")
        logger.info(f"self.num_workers: {self.num_workers}")
        logger.info(f"self.multi_gpu: {self.multi_gpu}")
        logger.info(f"self.use_cpu: {self.use_cpu}")
        logger.info(f"Using device: {self.device}")
        logger.info('-'*100)

    def perform_inference(self):
        if self.model_selection == 'pytesseract_en':
            # Perform OCR inference in English
            pass

        elif self.model_selection == 'pick':
            logger.info(f"self.checkpoint: {self.checkpoint}")
            logger.info(f"self.bt: {self.bt}")
            logger.info(f"self.impt: {self.impt}")
            logger.info(f"self.bs: {self.bs}")
            logger.info(f"self.output: {self.output}")
            logger.info(f"self.num_workers: {self.num_workers}")
            logger.info(f"self.multi_gpu: {self.multi_gpu}")
            logger.info(f"self.use_cpu: {self.use_cpu}")

            pick_inference = PICKInference(self.checkpoint, 
                                           self.bt, 
                                           self.impt, 
                                           self.bs, 
                                           self.output,
                                           self.num_workers,
                                           self.multi_gpu,
                                           self.use_cpu,
                                           debug_mode=self.debug_mode)  # Pass debug_mode here

            if self.use_cpu:
                logger.info("Using CPU for inference")
                pick_inference.cpu_call()
            elif self.multi_gpu:
                logger.info("Using multi-GPU for inference")
                pick_inference.spawn_processes()
            else:
                logger.info("Using single GPU for inference")
                pick_inference.perform_single_gpu_inference()

        elif self.model_selection == 'deepke':
            # Perform DeepKE inference
            pass

        elif self.model_selection == 'lilt':
            # Perform LiLT-Token Label Inference
            pass

        else:
            logger.error(f"Unsupported model selection: {self.model_selection}")

    def infer_single_image(self, image_path, boxes_and_transcripts):
        if self.model_selection == 'pick':
            pick_inference = PICKInference(checkpoint=self.checkpoint,
                                           bt=self.bt,
                                           impt=self.impt,
                                           bs=1,
                                           output=self.output,
                                           num_workers=self.num_workers,
                                           multi_gpu=self.multi_gpu,
                                           use_cpu=self.use_cpu,
                                           debug_mode=self.debug_mode)  # Pass debug_mode here
            return pick_inference.infer_single_image(image_path, boxes_and_transcripts)
        elif self.model_selection == 'pytesseract_en':
            # Implement single image inference for pytesseract
            pass
        elif self.model_selection == 'deepke':
            # Implement single image inference for deepke
            pass
        elif self.model_selection == 'lilt':
            # Implement single image inference for lilt
            pass
        else:
            logger.error(f"Unsupported model selection for single image inference: {self.model_selection}")
            return None

