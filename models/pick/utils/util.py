# -*- coding: utf-8 -*-

from typing import *
import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import torch

from .class_utils import keys_vocab_cls, iob_labels_vocab_cls
from data_utils import documents

import logging
# Configure the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def iob2entity(tag):
    '''
    iob label to entity
    :param tag:
    :return:
    '''
    if len(tag) == 1 and tag != 'O':
        raise TypeError('Invalid tag!')
    elif len(tag) == 1 and tag == 'O':
        return tag
    elif len(tag) > 1:
        e = tag[2:]
        return e

def iob_index_to_str(tags: List[List[int]]):
    decoded_tags_list = []
    for doc in tags:
        decoded_tags = []
        for tag in doc:
            s = iob_labels_vocab_cls.itos[tag]
            if s == '<unk>' or s == '<pad>':
                s = 'O'
            decoded_tags.append(s)
        decoded_tags_list.append(decoded_tags)
    return decoded_tags_list

def text_index_to_str(texts: torch.Tensor, mask: torch.Tensor):
    # union_texts: (B, num_boxes * T)
    union_texts = texts_to_union_texts(texts, mask)
    B, NT = union_texts.shape

    decoded_tags_list = []
    for i in range(B):
        decoded_text = []
        for text_index in union_texts[i]:
            text_str = keys_vocab_cls.itos[text_index]
            if text_str == '<unk>' or text_str == '<pad>':
                text_str = 'O'
            decoded_text.append(text_str)
        decoded_tags_list.append(decoded_text)
    return decoded_tags_list

def texts_to_union_texts(texts, mask):
    '''
    :param texts: (B, N, T)
    :param mask: (B, N, T)
    :return:
    '''

    B, N, T = texts.shape

    texts = texts.reshape(B, N * T)
    mask = mask.reshape(B, N * T)

    # union tags as a whole sequence, (B, N*T)
    union_texts = torch.full_like(texts, keys_vocab_cls['<pad>'], device=texts.device)

    max_seq_length = 0
    for i in range(B):
        valid_text = torch.masked_select(texts[i], mask[i].bool())
        valid_length = valid_text.size(0)
        union_texts[i, :valid_length] = valid_text

        if valid_length > max_seq_length:
            max_seq_length = valid_length

    # max_seq_length = documents.MAX_BOXES_NUM * documents.MAX_TRANSCRIPT_LEN
    # (B, N*T)
    union_texts = union_texts[:, :max_seq_length]

    # (B, N*T)
    return union_texts

def iob_tags_to_union_iob_tags(iob_tags, mask):
    '''
    :param iob_tags: (B, N, T)
    :param mask: (B, N, T)
    :return:
    '''

    B, N, T = iob_tags.shape

    iob_tags = iob_tags.reshape(B, N * T)
    mask = mask.reshape(B, N * T)

    # union tags as a whole sequence, (B, N*T)
    union_iob_tags = torch.full_like(iob_tags, iob_labels_vocab_cls['<pad>'], device=iob_tags.device)

    max_seq_length = 0
    for i in range(B):
        valid_tag = torch.masked_select(iob_tags[i], mask[i].bool())
        valid_length = valid_tag.size(0)
        union_iob_tags[i, :valid_length] = valid_tag

        if valid_length > max_seq_length:
            max_seq_length = valid_length

    # max_seq_length = documents.MAX_BOXES_NUM * documents.MAX_TRANSCRIPT_LEN
    # (B, N*T)
    union_iob_tags = union_iob_tags[:, :max_seq_length]

    # (B, N*T)
    return union_iob_tags

# Note: Entities without their positions to use in the inference code
def extract_entities(decoded_tags, decoded_texts):
    entities = []
    current_entity = None
    current_text = []

    for tag, char in zip(decoded_tags, decoded_texts):
        if tag.startswith("B-"):
            # If there's an ongoing entity, save it before starting a new one
            if current_entity is not None:
                entities.append({'entity_name': current_entity, 'text': ''.join(current_text)})

            # Start a new entity
            current_entity = tag[2:]  # Remove the "B-" prefix
            # print('current_entity',current_entity)
            current_text = [char]
            # print('current_test:', current_text)
        elif tag.startswith("I-") and current_entity is not None and tag[2:] == current_entity:
            # Continue the current entity
            current_text.append(char)
            # print('current_text: ',current_text)
        else:
            # If we hit an "O" or a different tag, finalize the current entity
            if current_entity is not None:
                entities.append({'entity_name': current_entity, 'text': ''.join(current_text)})
                # print('in for loop, ', 'entity_name: ',current_entity,'text: ',current_text)
                current_entity = None
                current_text = []
            

    # Add the last entity if there is one
    if current_entity is not None:
        entities.append({'entity_name': current_entity, 'text': ''.join(current_text)})
        # print('All done!',' entity_name: ',current_entity,'text: ',current_text)

    return entities

# Note: Entities with their positions to use in the inference code
def merge_boxes(boxes):
    """Merge multiple bounding boxes into one encompassing box."""
    x_coords = [coord for box in boxes for coord in [box[0][0], box[2][0]]]
    y_coords = [coord for box in boxes for coord in [box[0][1], box[2][1]]]
    return [
        (min(x_coords), min(y_coords)),  # top-left
        (max(x_coords), min(y_coords)),  # top-right
        (max(x_coords), max(y_coords)),  # bottom-right
        (min(x_coords), max(y_coords))   # bottom-left
    ]

def extract_entities_with_positions(decoded_tags, decoded_texts, boxes):
    entities = []
    current_entity = None
    current_text = []
    current_box = None
    word_idx = 0

    logger.info("Starting entity extraction with positions")

    for idx, (tag, char) in enumerate(zip(decoded_tags, decoded_texts)):
        logger.info(f"Processing idx: {idx}, tag: {tag}, char: {char}")

        if tag.startswith("B-"):
            # If there's an ongoing entity, save it before starting a new one
            if current_entity is not None:
                logger.info(f"Finalizing entity: {current_entity}, text: {''.join(current_text)}, position: {current_box}")
                entities.append({
                    'entity_name': current_entity, 
                    'text': ''.join(current_text),
                    'position': current_box
                })
                word_idx += 1

            # Start a new entity
            current_entity = tag[2:]  # Remove the "B-" prefix
            current_text = [char]
            current_box = boxes[word_idx]
            logger.info(f"Started new entity: {current_entity}")

        elif tag.startswith("I-") and current_entity is not None and tag[2:] == current_entity:
            # Continue the current entity
            current_text.append(char)
            logger.info(f"Continuing entity: {current_entity}, text so far: {''.join(current_text)}")

        else:
            # If we hit a different tag, finalize the current entity
            if current_entity is not None:
                logger.info(f"Finalizing entity: {current_entity}, text: {''.join(current_text)}, position: {current_box}")
                entities.append({
                    'entity_name': current_entity, 
                    'text': ''.join(current_text),
                    'position': current_box
                })
                current_entity = None
                current_text = []
                current_box = None
                word_idx += 1

    # Add the last entity if there is one
    if current_entity is not None:
        logger.info(f"Finalizing last entity: {current_entity}, text: {''.join(current_text)}, position: {current_box}")
        entities.append({
            'entity_name': current_entity, 
            'text': ''.join(current_text),
            'position': current_box
        })

    logger.info("Entity extraction complete")
    logger.info(f"Extracted entities: {entities}")

    return entities


