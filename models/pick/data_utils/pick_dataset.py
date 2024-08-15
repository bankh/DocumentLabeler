# # -*- coding: utf-8 -*-
# # @Author: Wenwen Yu
# # @Created Time: 7/9/2020 9:16 PM
import glob
import os
from typing import *
from pathlib import Path
import warnings
import random
from overrides import overrides

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

from . import documents
from .documents import MAX_BOXES_NUM, Document
from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls, entities_vocab_cls

class PICKDataset(Dataset):
    def __init__(self, files_name: str = None,
                 boxes_and_transcripts_folder: str = 'boxes_and_transcripts',
                 images_folder: str = 'images',
                 entities_folder: str = 'entities',
                 iob_tagging_type: str = 'box_and_within_box_level',
                 resized_image_size: Tuple[int, int] = (480, 960),
                 keep_ratio: bool = True,
                 ignore_error: bool = False,
                 training: bool = True):
        super().__init__()
        self._image_ext = None
        self._ann_ext = None
        self.iob_tagging_type = iob_tagging_type
        self.keep_ratio = keep_ratio
        self.ignore_error = ignore_error
        self.training = training
        assert resized_image_size and len(resized_image_size) == 2, 'resized image size not be set.'
        self.resized_image_size = tuple(resized_image_size)  # (w, h)
        self.max_boxes_num = MAX_BOXES_NUM

        if self.training:  # used for train and validation mode
            self.files_name = Path(files_name) if files_name else None
            self.data_root = self.files_name.parent if self.files_name else None
            self.boxes_and_transcripts_folder: Path = self.data_root.joinpath(boxes_and_transcripts_folder) if self.data_root else None
            self.images_folder: Path = self.data_root.joinpath(images_folder) if self.data_root else None
            self.entities_folder: Path = self.data_root.joinpath(entities_folder) if self.data_root else None
            if self.iob_tagging_type != 'box_level':
                if not self.entities_folder or not self.entities_folder.exists():
                    raise FileNotFoundError('Entity folder is not exist!')
        else:  # used for test mode
            self.boxes_and_transcripts_folder: Path = Path(boxes_and_transcripts_folder) if boxes_and_transcripts_folder else None
            self.images_folder: Path = Path(images_folder) if images_folder else None

        if self.boxes_and_transcripts_folder and self.images_folder:
            if not (self.boxes_and_transcripts_folder.exists() and self.images_folder.exists()):
                raise FileNotFoundError('Not contain boxes_and_transcripts folder {} or images folder {}.'
                                        .format(self.boxes_and_transcripts_folder, self.images_folder))
        
        self.files_list = []
        self.boxes_and_transcripts_list = []

        
        if self.training:
            self.files_list = pd.read_csv(self.files_name.as_posix(), header=None,
                                          names=['index', 'document_class', 'file_name'],
                                          dtype={'index': int, 'document_class': str, 'file_name': str})
        else:
            self.files_list = list(self.boxes_and_transcripts_folder.glob('*.tsv'))

    def __len__(self):
        return len(self.files_list)

    def get_image_file(self, basename):
        if self._image_ext is None:
            filename = list(self.images_folder.glob(f'**/{basename}.*'))[0]
            self._image_ext = os.path.splitext(filename)[1]
            print(f"Image file found: {filename}, extension: {self._image_ext}")
        return self.images_folder.joinpath(basename + self._image_ext)

    def get_ann_file(self, basename):
        if self._ann_ext is None:
            filename = list(self.boxes_and_transcripts_folder.glob(f'**/{basename}.*'))[0]
            self._ann_ext = os.path.splitext(filename)[1]
            print(f"Annotation file found: {filename}, extension: {self._ann_ext}")
        return self.boxes_and_transcripts_folder.joinpath(basename + self._ann_ext)


    @overrides
    def __getitem__(self, index):
        if self.training:
            dataitem: pd.Series = self.files_list.iloc[index]
            boxes_and_transcripts_file = self.get_ann_file(Path(dataitem['file_name']).stem)
            image_file = self.get_image_file(Path(dataitem['file_name']).stem)
            entities_file = self.entities_folder.joinpath(Path(dataitem['file_name']).stem + '.txt')
        else:
            boxes_and_transcripts_file = self.get_ann_file(Path(self.files_list[index]).stem)
            image_file = self.get_image_file(Path(self.files_list[index]).stem)

        # Check if the files exist and are not empty
        if not boxes_and_transcripts_file.exists() or not image_file.exists():
            if self.ignore_error:
                warnings.warn(f'{boxes_and_transcripts_file} or {image_file} does not exist. Getting a new one.')
                new_item = random.randint(0, len(self) - 1)
                return self.__getitem__(new_item)
            else:
                raise RuntimeError(f'Sample: {boxes_and_transcripts_file.stem} does not exist.')

        if boxes_and_transcripts_file.stat().st_size == 0 or image_file.stat().st_size == 0:
            if self.ignore_error:
                warnings.warn(f"Warning: Skipping empty file {boxes_and_transcripts_file} or {image_file}. Generating a new item.")
                new_item = random.randint(0, len(self) - 1)
                return self.__getitem__(new_item)
            else:
                raise RuntimeError(f"Error: File {boxes_and_transcripts_file} or {image_file} is empty.")
        
        try:
            if self.training:
                document = Document(boxes_and_transcripts_file, 
                                    image_file, 
                                    self.resized_image_size,
                                    self.iob_tagging_type, 
                                    entities_file, 
                                    training=self.training,
                                    max_boxes_num=self.max_boxes_num)
            else:
                document = Document(boxes_and_transcripts_file, 
                                    image_file, 
                                    self.resized_image_size,
                                    image_index=index, 
                                    training=self.training,
                                    max_boxes_num=self.max_boxes_num)
            
            if hasattr(document, 'chunks'):
                return document.chunks
            else:
                return [document]
            
        except Exception as e:
            if self.ignore_error:
                warnings.warn(f'Error loading sample: {e}. Generating a new item.')
                new_item = random.randint(0, len(self) - 1)
                return self.__getitem__(new_item)
            else:
                raise RuntimeError(f'Error occurs in image {boxes_and_transcripts_file.stem}: {e}')

class BatchCollateFn(object):
    def __init__(self, training: bool = True):
        self.trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.training = training
   
    def __call__(self, batch_list: List[Document]):
        batch_list = [item for sublist in batch_list for item in (sublist if isinstance(sublist, list) else [sublist])]
                
        if not batch_list:
            raise ValueError("The batch_list is empty after filtering out None values.")

        max_boxes_num_batch = max([x.boxes_num for x in batch_list])
        max_transcript_len = max([x.transcript_len for x in batch_list])

        image_batch_tensor = torch.stack([self.trsfm(x.whole_image) for x in batch_list], dim=0).float()

        relation_features_padded_list = [F.pad(torch.FloatTensor(x.relation_features),
                                               (0,0,0,max_boxes_num_batch - x.boxes_num,0,max_boxes_num_batch - x.boxes_num))
                                         for i, x in enumerate(batch_list)]
        relation_features_batch_tensor = torch.stack(relation_features_padded_list, dim=0)  

        boxes_coordinate_padded_list = [F.pad(torch.FloatTensor(x.boxes_coordinate),
                                              (0,0,0,max_boxes_num_batch - x.boxes_num))
                                        for i, x in enumerate(batch_list)]
        boxes_coordinate_batch_tensor = torch.stack(boxes_coordinate_padded_list, dim=0)

        text_segments_padded_list = [F.pad(torch.LongTensor(x.text_segments[0]),
                                           (0, 
                                            max_transcript_len - x.transcript_len,
                                            0, 
                                            max_boxes_num_batch - x.boxes_num),
                                           value=keys_vocab_cls.stoi['<pad>'])
                                     for i, x in enumerate(batch_list)]
        text_segments_batch_tensor = torch.stack(text_segments_padded_list, dim=0)

        text_length_padded_list = [F.pad(torch.LongTensor(x.text_segments[1]),
                                         (0, 
                                          max_boxes_num_batch - x.boxes_num))
                                   for i, x in enumerate(batch_list)]
        text_length_batch_tensor = torch.stack(text_length_padded_list, dim=0)

        mask_padded_list = [F.pad(torch.ByteTensor(x.mask),
                                  (0, 
                                   max_transcript_len - x.transcript_len,
                                   0, 
                                   max_boxes_num_batch - x.boxes_num))
                            for i, x in enumerate(batch_list)]
        mask_batch_tensor = torch.stack(mask_padded_list, dim=0)

        if self.training:
            iob_tags_label_padded_list = [F.pad(torch.LongTensor(x.iob_tags_label),
                                                (0, 
                                                 max_transcript_len - x.transcript_len,
                                                 0, 
                                                 max_boxes_num_batch - x.boxes_num),
                                                 value=iob_labels_vocab_cls.stoi['<pad>'])
                                          for i, x in enumerate(batch_list)]
            iob_tags_label_batch_tensor = torch.stack(iob_tags_label_padded_list, dim=0)

        else:
            image_indexs_list = [x.image_index for x in batch_list]
            image_indexs_tensor = torch.tensor(image_indexs_list)

        filenames = [doc.image_filename for doc in batch_list]

        if self.training:
            batch = dict(whole_image=image_batch_tensor,
                         relation_features=relation_features_batch_tensor,
                         text_segments=text_segments_batch_tensor,
                         text_length=text_length_batch_tensor,
                         boxes_coordinate=boxes_coordinate_batch_tensor,
                         mask=mask_batch_tensor,
                         iob_tags_label=iob_tags_label_batch_tensor,
                         filenames=filenames)
        else:
            batch = dict(whole_image=image_batch_tensor,
                         relation_features=relation_features_batch_tensor,
                         text_segments=text_segments_batch_tensor,
                         text_length=text_length_batch_tensor,
                         boxes_coordinate=boxes_coordinate_batch_tensor,
                         mask=mask_batch_tensor,
                         image_indexs=image_indexs_tensor,
                         filenames=filenames)

        return batch
