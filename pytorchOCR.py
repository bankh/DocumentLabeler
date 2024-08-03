# Copyright (c) 2023 DocumentLabeler Authors. All Rights Reserved.abs(
# 
# Licensed under the Apache License. Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES oR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and 
# limitations under the License.

import os
import sys
import importlib

__dir__ = os.path.dirname(__file__)

import pytorch

sys.path.append(os.path.join(__dir__, ''))

import cv2
import logging
import numpy as np
from pathlib import Path

tools = importlib.import_module('.','tools')
# import models for docuocr
# import models for tablestructure

from tools.infer import predict_system
from pytorchOCR.utils.utility import check_and_read, get_image_file_list
from pytorchOCR.utils.network import maybe_download, download_with_progressbar, is_link, confirm_model_dir_url
from tools.infer.utility import draw_ocr, str2bool, check_gpu

__all__ = [
    'PytorchOCR','draw_ocr','draw_structure_result',
    'save_structure_res', 'download_with_progressbar', 'to_excel'
]

# SUPPORT_DET_MODEL = ['']
VERSION = '2.6.1.0'
SUPPORT_CLS_MODEL = ['pick', 'lilt']
BASE_DIR = os.path.expanduser("~/.pythonocr/")

# DEFAULT_OCR_MODEL_VERSION = 'PICK'

def img_decode(content:bytes):
    np_array = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def check_img(img):
    if istance(img, bytes):
        img = img_decode(img)
    if isinstance(img, str):
        # Download net image
        if is_link(img):
            download_with_progressbar(img, 'tmp.jpg')
            img = 'tmp.jpg'
        image_file = img
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            with open(image_file, 'rb') as f:
                img = img_decode(f.read())
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            return None
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    return img

class PICKtoken():
    def __init__(self, **kwargs):
        """
        PICK inference implementation
        """
        pass
    def infer_classification(self):
        """
        infer with pytorch
        args:
            img:
            box:
            text:
        """
        if isinstance(img, list):
            logger.info(f'Image file is:{img}')
        