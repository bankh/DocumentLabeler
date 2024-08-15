import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from torchtext.data import Field, RawField
from torchvision import transforms

import cv2
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
import numpy as np
import logging

from pathlib import Path
from tqdm import tqdm
import os
import sys
from typing import *

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import model.pick as pick_arch_module
# from data_utils import documents
from data_utils.documents import MAX_BOXES_NUM, Document
from utils.class_utils import keys_vocab_cls

from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str, extract_entities_with_positions #extract_entities
from PIL import Image

logger = logging.getLogger(__name__)

def convert_scores_to_probabilities(scores):
    scores = np.array(scores)
    probabilities = np.exp(scores) / np.sum(np.exp(scores))
    return probabilities.tolist()

class PICKInference:
    def __init__(self, 
                 checkpoint, 
                 bt, 
                 impt, 
                 bs=16, 
                 output='output', 
                 num_workers=2,
                 multi_gpu=False,
                 use_cpu=False,
                 debug_mode=False):
        self.checkpoint = checkpoint
        self.bt = bt
        self.impt = impt
        self.bs = bs
        self.output = output
        self.num_workers = num_workers
        self.multi_gpu = multi_gpu
        self.debug_mode = debug_mode
        
        self.use_cpu = use_cpu
        self.device = torch.device("cpu" if use_cpu else "cuda" if torch.cuda.is_available() else "cpu")
        
        if self.multi_gpu:
            os.environ['NCCL_DEBUG'] = 'INFO'
        
        self.resized_image_size = (480, 960)  # You might want to make this configurable

    def spawn_processes(self):
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(self.run_distributed, args=(world_size,), nprocs=world_size, join=True)
    
    def run_distributed(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        self.__call__(rank)

    def __call__(self, rank):
        if torch.cuda.is_available():
            self.multi_gpu_call(rank)
        else:
            self.cpu_call()

    def multi_gpu_call(self, rank):
        device = torch.device("cuda", rank)
        if self.debug_mode:
            logger.info(f'Using GPU: {device} for inference.')
            logger.info(f'Environment Variables: {os.environ}')

        try:
            torch.distributed.init_process_group(backend="nccl")
        except Exception as e:
            logger.error(f"Failed to initialize process group: {e}")
            return

        checkpoint = torch.load(self.checkpoint, map_location=device)
        config = checkpoint['config']
        state_dict = checkpoint['state_dict']
        monitor_best = checkpoint['monitor_best']
        if self.debug_mode:
            logger.info(f'Loading checkpoint: {self.checkpoint} with saved mEF {monitor_best:.4f} ...')

        pick_model = config.init_obj('model_arch', pick_arch_module)
        pick_model = pick_model.to(device)
        pick_model.load_state_dict(state_dict)

        pick_model = DDP(pick_model, device_ids=[rank], output_device=rank)
        pick_model.eval()

        infer_dataset = PICKDataset(boxes_and_transcripts_folder=self.bt,
                                    images_folder=self.impt,
                                    resized_image_size=(480, 960),
                                    ignore_error=False,
                                    training=False)
        infer_sampler = DistributedSampler(infer_dataset)
        infer_data_loader = DataLoader(infer_dataset, 
                                       batch_size=self.bs, 
                                       shuffle=False,
                                       num_workers=self.num_workers, 
                                       collate_fn=BatchCollateFn(training=False),
                                       sampler=infer_sampler)

        output_path = Path(self.output)
        output_path.mkdir(parents=True, exist_ok=True)

        self.process_batch(pick_model, infer_data_loader, device, output_path, infer_dataset)
        if self.debug_mode:
            logger.info(f'GPU {rank} finished inference.')
        torch.distributed.destroy_process_group()

    def perform_single_gpu_inference(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.debug_mode:
            logger.info(f'Using device: {device} for inference')
            logger.info(f'Checkpoint path: {self.checkpoint}')

        if not self.checkpoint or not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found at path: {self.checkpoint}")

        checkpoint = torch.load(self.checkpoint, map_location=device)
        config = checkpoint['config']
        state_dict = checkpoint['state_dict']
        monitor_best = checkpoint['monitor_best']
        if self.debug_mode:
            logger.info(f'Loading checkpoint: {self.checkpoint} with saved mEF {monitor_best:.4f} ...')

        pick_model = config.init_obj('model_arch', pick_arch_module)
        pick_model = pick_model.to(device)
        pick_model.load_state_dict(state_dict)
        pick_model.eval()

        infer_dataset = PICKDataset(boxes_and_transcripts_folder=self.bt,
                                    images_folder=self.impt,
                                    resized_image_size=(480, 960),
                                    ignore_error=False,
                                    training=False)
        infer_data_loader = DataLoader(infer_dataset,
                                       batch_size=self.bs,
                                       shuffle=False,
                                       num_workers=self.num_workers,
                                       collate_fn=BatchCollateFn(training=False))

        output_path = Path(self.output)
        output_path.mkdir(parents=True, exist_ok=True)

        self.process_batch(pick_model, infer_data_loader, device, output_path, infer_dataset)
        if self.debug_mode:
            logger.info('Single GPU inference finished.')

    def cpu_call(self):
        if self.debug_mode:
            logger.info('Using CPU for inference.')
        device = torch.device("cpu")

        checkpoint = torch.load(self.checkpoint, map_location=device)
        config = checkpoint['config']
        state_dict = checkpoint['state_dict']
        monitor_best = checkpoint['monitor_best']
        if self.debug_mode:
            logger.info(f'Loading checkpoint: {self.checkpoint} with saved mEF {monitor_best:.4f} ...')

        pick_model = config.init_obj('model_arch', pick_arch_module)
        pick_model = pick_model.to(device)
        pick_model.load_state_dict(state_dict)
        pick_model.eval()

        infer_dataset = PICKDataset(boxes_and_transcripts_folder=self.bt,
                                    images_folder=self.impt,
                                    resized_image_size=(480, 960),
                                    ignore_error=False,
                                    training=False)
        infer_data_loader = DataLoader(infer_dataset, 
                                       batch_size=self.bs, 
                                       shuffle=False,
                                       num_workers=self.num_workers, 
                                       collate_fn=BatchCollateFn(training=False))

        output_path = Path(self.output)
        output_path.mkdir(parents=True, exist_ok=True)

        self.process_batch(pick_model, infer_data_loader, device, output_path, infer_dataset)
        if self.debug_mode:
            logger.info('CPU inference finished.')

    def process_batch(self, pick_model, data_loader, device, output_path, dataset):
        with torch.no_grad():
            for step_idx, input_data_item in tqdm(enumerate(data_loader)):
                if input_data_item is None:  # Skip if data item is None
                    continue
        
                for key, input_value in input_data_item.items():
                    if input_value is not None and isinstance(input_value, torch.Tensor):
                        input_data_item[key] = input_value.to(device)

                # Check the extracted content if in debug mode
                if self.debug_mode:
                    # Debug print statements to check tensor shapes
                    print("\n")
                    logger.info(f"Batch {step_idx + 1}:")

                    logger.info(f"Shape of whole_image: {input_data_item['whole_image'].shape}")
                    # logger.info(f"Content of whole_image: {input_data_item['whole_image']}")

                    logger.info(f"Shape of boxes_coordinate: {input_data_item['boxes_coordinate'].shape}")
                    # logger.info(f"Content of boxes_coordinate: {input_data_item['boxes_coordinate']})

                    logger.info(f"Shape of text_segments: {input_data_item['text_segments'].shape}")
                    # logger.info(f"Content of text_segments: {input_data_item['text_segments']}")

                    logger.info(f"Shape of mask: {input_data_item['mask'].shape}")
                    # logger.info(f"Content of mask: {input_data_item['mask']}")

                    logger.info(f"Shape of relation_features: {input_data_item['relation_features'].shape}")
                    # logger.info(f"Content of relation_features: {input_data_item['relation_features']}")

                    print("-" * 50)

                output = pick_model(**input_data_item)
                logits = output['logits']
                new_mask = output['new_mask']
                image_indexs = input_data_item['image_indexs']
                text_segments = input_data_item['text_segments']
                mask = input_data_item['mask']

                best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits,
                                                                       mask=new_mask,
                                                                       logits_batch_first=True)
                predicted_tags = [path for path, _ in best_paths]

                decoded_tags_list = iob_index_to_str(predicted_tags)
                decoded_texts_list = text_index_to_str(text_segments, mask)

                for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list,
                                                                    decoded_texts_list,
                                                                    image_indexs):
                    spans = bio_tags_to_spans(decoded_tags, [])
                    spans = sorted(spans, key=lambda x: x[1][0])

                    entities = []
                    for entity_name, range_tuple in spans:
                        entity = dict(entity_name=entity_name,
                                      text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                        entities.append(entity)

                    result_file = output_path.joinpath(Path(dataset.files_list[image_index]).stem + '.txt')

                    with result_file.open(mode='w') as f:
                        for item in entities:
                            f.write(f'{item["entity_name"]}\t{item["text"]}\n')
    
    def infer_single_image(self, image_path, boxes_and_transcripts, image_index=None):
        checkpoint = torch.load(self.checkpoint, map_location=self.device)
        config = checkpoint['config']
        state_dict = checkpoint['state_dict']

        # Loading and preparing the model
        pick_model = config.init_obj('model_arch', pick_arch_module)
        pick_model = pick_model.to(self.device)
        pick_model.load_state_dict(state_dict)
        pick_model.eval()

        MAX_BOXES_NUM = 1000
        MAX_TRANSCRIPT_LEN = 30  # Adjust as necessary

        TextSegmentsField = Field(sequential=True, use_vocab=True, include_lengths=True, batch_first=True)
        TextSegmentsField.vocab = keys_vocab_cls  # Ensure keys_vocab_cls is defined

        if not isinstance(image_path, (str, Path)):
            raise TypeError("image_path must be a string or Path object.")
            
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} does not exist.")
        if not image_path.is_file():
            raise IOError(f"{image_path} is not a file.")

        image = cv2.imread(str(image_path))
        if image is None:
            raise IOError(f"Failed to load image {image_path}")

        height, width, _ = image.shape

        # Resize image
        image = cv2.resize(image, self.resized_image_size, interpolation=cv2.INTER_LINEAR)
        x_scale = self.resized_image_size[0] / width
        y_scale = self.resized_image_size[1] / height

        # Convert image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transpose the image to (C, H, W) format and normalize
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Convert to tensor and add batch dimension
        self.whole_image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        if self.debug_mode:
            logger.info(f"Shape of whole_image: {self.whole_image.shape}")

        # Prepare boxes and transcripts data without flattening
        formatted_boxes_and_transcripts = []
        for i, (box, transcript) in enumerate(boxes_and_transcripts):
            if len(box) != 4 or any(len(point) != 2 for point in box):
                raise ValueError(f"Box at index {i} does not have the correct format. Expected 4 tuples of 2 coordinates each.")
            formatted_boxes_and_transcripts.append((i, box, transcript))

        # Sort the boxes based on the position using the same method as in Document.py
        sorted_boxes_and_transcripts = self.sort_box_with_list(formatted_boxes_and_transcripts)

        boxes, transcripts = [], []

        for _, points, transcript in sorted_boxes_and_transcripts:
            if len(transcript) == 0:
                transcript = ' '
            boxes.append(points)
            transcripts.append(transcript)

        # Limit the number of boxes and transcripts to process, following Document.py
        boxes_num = min(len(boxes), MAX_BOXES_NUM)
        transcript_len = min(max([len(t) for t in transcripts[:boxes_num]]), MAX_TRANSCRIPT_LEN)

        # Create the mask and relation_features arrays
        mask = np.zeros((boxes_num, transcript_len),dtype=int)
        relation_features = np.zeros((boxes_num, boxes_num, 6))

        # Get min area box for each (original) box to calculate initial relation features
        min_area_boxes = [cv2.minAreaRect(np.array(box, dtype=np.float32).reshape(4, 2)) for box in boxes[:boxes_num]]

        # Calculate resized image box coordinates and initial relation features between boxes (nodes)
        resized_boxes = []
        for i in range(boxes_num):
            box_i = boxes[i]
            transcript_i = transcripts[i]

            # Get resized image box coordinates, used for ROIAlign in Encoder layer
            resized_box_i = []
            for j, pos in enumerate(box_i):
                if isinstance(pos, (tuple, list)) and len(pos) == 2:
                    x, y = pos
                    scaled_x = int(np.round(x * x_scale))
                    scaled_y = int(np.round(y * y_scale))
                    resized_box_i.extend([scaled_x, scaled_y])
                else:
                    raise TypeError(f"Expected a tuple of two coordinates but got {type(pos)}: {pos}")

            resized_rect_output_i = cv2.minAreaRect(np.array(resized_box_i, dtype=np.float32).reshape(4, 2))
            resized_box_i = cv2.boxPoints(resized_rect_output_i)
            resized_box_i = resized_box_i.reshape((8,))
            resized_boxes.append(resized_box_i)

            # Enumerate each box and calculate relation features between i and other nodes
            self.relation_features_between_ij_nodes(boxes_num, i, min_area_boxes, relation_features, transcript_i, transcripts)

        # Normalize the relation features
        relation_features = self.normalize_relation_features(relation_features, width=width, height=height)
        if not isinstance(relation_features, torch.Tensor):
            relation_features = torch.tensor(relation_features, dtype=torch.float32).to(self.device)

        # Process the transcripts using the exact approach as in Document.py
        text_segments = [list(trans) for trans in transcripts[:boxes_num]]
        texts, texts_len = TextSegmentsField.process(text_segments)
        texts = texts[:, :transcript_len].numpy()
        texts_len = np.clip(texts_len.numpy(), 0, transcript_len)
        text_segments_tensor = torch.tensor(texts).to(self.device)  # Convert texts to tensor
        text_length_tensor = torch.tensor(texts_len).to(self.device)

        # Add batch dimension to text_segments_tensor if it's missing
        if text_segments_tensor.dim() == 2:
            text_segments_tensor = text_segments_tensor.unsqueeze(0)  # Add batch dimension
            text_length_tensor = text_length_tensor.unsqueeze(0)  # Add batch dimension to text_length_tensor as well
        
        if self.debug_mode:
            logger.info(f"Shape of text_segments_tensor: {text_segments_tensor.shape}")
            logger.info(f"Shape of text_length_tensor: {text_length_tensor.shape}")

        # Efficiently fill the mask with 1's based on the transcript lengths
        for i in range(boxes_num):
            mask[i, :texts_len[i]] = 1  # Set the first `texts_len[i]` elements to 1
        # Add batch dimension to mask
        mask = np.expand_dims(mask, axis=0)  # Add batch dimension (1, N, T)
        mask = torch.tensor(mask, dtype=torch.uint8).to(self.device)

        self.boxes_coordinate = RawField().preprocess(resized_boxes)
        self.relation_features = RawField().preprocess(relation_features)
        self.mask = RawField().preprocess(mask)
        self.boxes_num = RawField().preprocess(boxes_num)
        self.transcript_len = RawField().preprocess(transcript_len)
        self.image_index = RawField().preprocess(image_index)

        # Prepare the input data dictionary
        input_data = {
            'whole_image': self.whole_image,
            'boxes_coordinate': self.boxes_coordinate,
            'text_segments': text_segments_tensor,  # Use the tensor directly
            'mask': self.mask,  # Ensure the mask has 3 dimensions (Batch, N, T)
            'relation_features': self.relation_features,
            'boxes_num': self.boxes_num,
            'transcript_len': self.transcript_len,
            'image_index': self.image_index,
            'text_length': text_length_tensor  # Ensure text_length is included
        }

        if self.debug_mode:
            # Add print statements to check shapes and full content
            logger.info(f"Shape of whole_image: {input_data['whole_image'].shape}")
            # logger.info(f"Content of whole_image: {input_data['whole_image']}")
            # logger.info(f"Shape of boxes_coordinate: {input_data['boxes_coordinate'].shape}")
            # logger.info(f"Content of boxes_coordinate: {input_data_item['boxes_coordinate']})
            logger.info(f"Shape of text_segments: {input_data['text_segments'].shape}")
            # logger.info(f"Content of text_segments: {input_data['text_segments']}")
            logger.info(f"Shape of mask: {input_data['mask'].shape}")
            # logger.info(f"Content of mask: {input_data['mask']}")
            logger.info(f"Shape of relation_features: {input_data['relation_features'].shape}")
            # logger.info(f"Content of relation_features: {input_data['relation_features']}")

        # Inference
        with torch.no_grad():
            if self.debug_mode:
                logger.info(f"Input Data Keys: {input_data.keys()}")
                logger.info(f"Mask Shape: {input_data['mask'].shape}")
            output = pick_model(**input_data)
            logits = output['logits']
            new_mask = output['new_mask']
            
            if self.debug_mode:
                logger.info(f"Shape of new_mask output: {new_mask.shape}")

            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, 
                                                                mask=new_mask, 
                                                                logits_batch_first=True)
            predicted_tags = best_paths[0][0]

            decoded_tags = iob_index_to_str([predicted_tags])[0]
            if self.debug_mode:
                logger.info(f"decoded_tags: {decoded_tags}")
                logger.info(f"Length of decoded_tags: {len(decoded_tags)}")
            decoded_texts = text_index_to_str(text_segments_tensor, mask)[0]
            if self.debug_mode:
                logger.info(f"decoded_texts: {decoded_texts}")
                logger.info(f"Length of decoded_text: {len(decoded_texts)}")
            
            entities = extract_entities_with_positions(decoded_tags, decoded_texts, boxes[:boxes_num])

            return entities
       
    def sort_box_with_list(self, data: List[Tuple], left_right_first=False):
        def compare_key(x):
            # x is (index, points, transcription)
            points = x[1]
            box = np.array([[points[0][0], points[0][1]], 
                            [points[1][0], points[1][1]], 
                            [points[2][0], points[2][1]], 
                            [points[3][0], points[3][1]]], dtype=np.float32)
            rect = cv2.minAreaRect(box)
            center = rect[0]
            if left_right_first:
                return center[0], center[1]
            else:
                return center[1], center[0]

        data = sorted(data, key=compare_key)
        return data

    def normalize_relation_features(self, feat: np.ndarray, width: int, height: int):
        np.clip(feat, 1e-8, np.inf)
        feat[:, :, 0] = feat[:, :, 0] / width
        feat[:, :, 1] = feat[:, :, 1] / height

        # The second graph to the 6th graph.
        for i in range(2, 6):
            feat_ij = feat[:, :, i]
            max_value = np.max(feat_ij)
            min_value = np.min(feat_ij)
            if max_value != min_value:
                feat[:, :, i] = feat[:, :, i] - min_value / (max_value - min_value)
        return feat

    def relation_features_between_ij_nodes(self, boxes_num, i, min_area_boxes, relation_features, transcript_i,
                                        transcripts):
        for j in range(boxes_num):
            transcript_j = transcripts[j]

            rect_output_i = min_area_boxes[i]
            rect_output_j = min_area_boxes[j]

            center_i = rect_output_i[0]
            center_j = rect_output_j[0]

            width_i, height_i = rect_output_i[1]
            width_j, height_j = rect_output_j[1]

        relation_features[i, j, 0] = np.abs(center_i[0] - center_j[0]) if np.abs(center_i[0] - center_j[0]) is not None else -1  # x_ij
        relation_features[i, j, 1] = np.abs(center_i[1] - center_j[1]) if np.abs(center_i[1] - center_j[1]) is not None else -1  # y_ij
        relation_features[i, j, 2] = width_i / (height_i) if height_i != 0 and width_i / (height_i) is not None else -1  # w_i/h_i
        relation_features[i, j, 3] = height_j / (height_i) if height_i != 0 and height_j / (height_i) is not None else -1  # h_j/h_i
        relation_features[i, j, 4] = width_j / (height_i) if height_i != 0 and width_j / (height_i) is not None else -1  # w_j/h_i
        relation_features[i, j, 5] = len(transcript_j) / (len(transcript_i)) if len(transcript_j) / (len(transcript_i)) is not None else -1  # T_j/T_i
