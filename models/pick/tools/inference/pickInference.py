import torch
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Process
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
import numpy as np

from pathlib import Path
from tqdm import tqdm
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# from parse_config import ConfigParser
# For debugging and command line check with the arguments
import argparse
import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str

def convert_scores_to_probabilities(scores):
    # Convert scores to numpy array for numerical stability
    scores = np.array(scores)

    # Apply softmax normalization
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
                 local_rank=0):
        self.checkpoint = checkpoint
        self.bt = bt
        self.impt = impt
        self.bs = bs
        self.output = output
        self.local_rank = local_rank
        self.num_workers = num_workers
    
    def spawn_processes(self):
        # Spawn processes
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(self.run_distributed, args=(world_size,), nprocs=world_size, join=True)
    
    def run_distributed(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        self.__call__(rank)                      

    def __call__(self,rank):
        if torch.cuda.is_available():
            self.multi_gpu_call(rank)
        else:
            device = torch.device("cpu")  # Assign the CPU device to 'device'
            self.cpu_call(device)

    # Run the inference code from GPU or multiple GPUs
    def multi_gpu_call(self, rank):
        device = torch.device("cuda", rank)  # Assign the CUDA device to 'device'
        print('Using GPU: {} for inference.'.format(device))

        init_process_group(backend="nccl")

        checkpoint = torch.load(self.checkpoint, map_location=device)

        config = checkpoint['config']
        state_dict = checkpoint['state_dict']
        monitor_best = checkpoint['monitor_best']
        print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(self.checkpoint, monitor_best))

        # prepare model for testing
        pick_model = config.init_obj('model_arch', pick_arch_module)
        pick_model = pick_model.to(device)
        pick_model.load_state_dict(state_dict)

        pick_model = DDP(pick_model, 
                         device_ids=[rank], 
                         output_device=rank)
        pick_model.eval()

        # setup dataset and data_loader instances
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

        # setup output path
        output_path = Path(self.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # predict and save to file
        with torch.no_grad():
            for step_idx, input_data_item in tqdm(enumerate(infer_data_loader)):
                for key, input_value in input_data_item.items():
                    if input_value is not None and isinstance(input_value, torch.Tensor):
                        input_data_item[key] = input_value.to(device)

                # For easier debug.
                image_names = input_data_item["filenames"]

                output = pick_model(**input_data_item)
                logits = output['logits']                           # (B, N*T, out_dim)
                new_mask = output['new_mask']
                image_indexs = input_data_item['image_indexs']      # (B,)
                text_segments = input_data_item['text_segments']    # (B, num_boxes, T)
                mask = input_data_item['mask']
                
                # List[(List[int], torch.Tensor)]
                best_paths = pick_model.module.decoder.crf_layer.viterbi_tags(logits, 
                                                                              mask=new_mask, 
                                                                              logits_batch_first=True)
                predicted_tags = []
                predicted_probabilities = []

                # Collect the scores from best_paths
                scores = [score for _, score in best_paths]

                # Convert scores to probabilities
                probabilities = convert_scores_to_probabilities(scores)

                for path, probability in zip(best_paths, probabilities):
                    predicted_tags.append(path[0])
                    predicted_probabilities.append(probability)

                    print("Path:", path[0])
                    print("Score:", path[1])
                    print("Probabilities:", probability)
                    print()

                # Print the text_segments
                # print('input_data_item:', input_data_item['text_segments'], '\n')

                # convert iob index to iob string
                decoded_tags_list = iob_index_to_str(predicted_tags)

                decoded_tags_list = iob_index_to_str(predicted_tags)

                for decoded_tags in decoded_tags_list:
                    print(decoded_tags)
                # union text as a sequence and convert index to string
                decoded_texts_list = text_index_to_str(text_segments, mask)

                for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, 
                                                                    decoded_texts_list, 
                                                                    image_indexs):
                    # List[ Tuple[str, Tuple[int, int]] ]
                    spans = bio_tags_to_spans(decoded_tags, [])
                    spans = sorted(spans, key=lambda x: x[1][0])

                    # Print the text_segments
                    # print('decoded_texts:', decoded_texts, '\n')

                    entities = []  # exists one to many case
                    for entity_name, range_tuple in spans:
                        entity = dict(entity_name=entity_name,
                                      text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                        entities.append(entity)

                    result_file = output_path.joinpath(Path(infer_dataset.files_list[image_index]).stem + '.txt')

                    # Print the result file path
                    print('\nSaving results to file: {}'.format(result_file))

                    with result_file.open(mode='w') as f:
                        for item in entities:
                            print('{}\t{}'.format(item['entity_name'], item['text']))  # Print the inferred text
                            f.write('{}\t{}\n'.format(item['entity_name'], item['text']))

        print('GPU {} finished inference.'.format(rank))
        destroy_process_group()

    # Run the inference code from CPU
    def cpu_call(self, device):
        print('Using CPU for inference.')
        checkpoint = torch.load(self.checkpoint, map_location=device)

        config = checkpoint['config']
        state_dict = checkpoint['state_dict']
        monitor_best = checkpoint['monitor_best']
        print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(self.checkpoint, monitor_best))

        # prepare model for testing
        pick_model = config.init_obj('model_arch', pick_arch_module)
        pick_model = pick_model.to(device)
        pick_model.load_state_dict(state_dict)
        pick_model.eval()

        # setup dataset and data_loader instances
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

        # setup output path
        output_path = Path(self.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # predict and save to file
        with torch.no_grad():
            for step_idx, input_data_item in tqdm(enumerate(infer_data_loader)):
                for key, input_value in input_data_item.items():
                    if input_value is not None and isinstance(input_value, torch.Tensor):
                        input_data_item[key] = input_value.to(device)

                # For easier debug.
                image_names = input_data_item["filenames"]

                output = pick_model(**input_data_item)
                logits = output['logits']                           # (B, N*T, out_dim)
                new_mask = output['new_mask']
                image_indexs = input_data_item['image_indexs']      # (B,)
                text_segments = input_data_item['text_segments']    # (B, num_boxes, T)
                mask = input_data_item['mask']
                # List[(List[int], torch.Tensor)]
                best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, 
                                                                       mask=new_mask, 
                                                                       logits_batch_first=True)
                
                predicted_tags = []
                for path, score in best_paths:
                    predicted_tags.append(path)

                # convert iob index to iob string
                decoded_tags_list = iob_index_to_str(predicted_tags)
                # union text as a sequence and convert index to string
                decoded_texts_list = text_index_to_str(text_segments, mask)

                for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
                    # List[ Tuple[str, Tuple[int, int]] ]
                    spans = bio_tags_to_spans(decoded_tags, [])
                    spans = sorted(spans, key=lambda x: x[1][0])

                    entities = []  # exists one to many case
                    for entity_name, range_tuple in spans:
                        entity = dict(entity_name=entity_name,
                                    text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                        entities.append(entity)

                    result_file = output_path.joinpath(Path(infer_dataset.files_list[image_index]).stem + '.txt')
                    with result_file.open(mode='w') as f:
                        for item in entities:
                            f.write('{}\t{}\n'.format(item['entity_name'], item['text']))

        print('CPU finished inference.')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Multi-GPU version of the original code')
#     parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
#     parser.add_argument('--bt', type=str, required=True, help='path to boxes and transcripts folder')
#     parser.add_argument('--impt', type=str, required=True, help='path to images folder')
#     parser.add_argument('--bs', type=int, default=16, help='batch size')
#     parser.add_argument('--output', type=str, default='output', help='output path')
#     parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
#     args = parser.parse_args()
#     inference = Inference(args.checkpoint, args.bt, args.impt, args.bs, args.output, args.local_rank)
    # inference.run_inference()
