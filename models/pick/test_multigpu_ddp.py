import torch
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from pathlib import Path
from tqdm import tqdm

from parse_config import ConfigParser
import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str

def group_entities(tokens, labels):
    entities = []
    current_entity_tokens = []
    current_label = None

    for token, label in zip(tokens, labels):
        if label == current_label:
            current_entity_tokens.append(token)
        else:
            if current_entity_tokens:
                entities.append((' '.join(current_entity_tokens), current_label))
                current_entity_tokens = []
            if label != "O":
                current_entity_tokens.append(token)
                current_label = label

    if current_entity_tokens:
        entities.append((' '.join(current_entity_tokens), current_label))

    return entities

def main(args):
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    init_process_group(backend="nccl")

    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    monitor_best = checkpoint['monitor_best']
    print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(args.checkpoint, monitor_best))

    # prepare model for testing
    pick_model = config.init_obj('model_arch', pick_arch_module)
    pick_model = pick_model.to(device)
    pick_model.load_state_dict(state_dict)

    pick_model = DDP(pick_model, device_ids=[local_rank], output_device=local_rank)

    pick_model.eval()

    # setup dataset and data_loader instances
    test_dataset = PICKDataset(boxes_and_transcripts_folder=args.bt,
                               images_folder=args.impt,
                               resized_image_size=(762, 1000),
                               ignore_error=False,
                               training=False)
    test_sampler = DistributedSampler(test_dataset)
    test_data_loader = DataLoader(test_dataset, 
                                  batch_size=args.bs, 
                                  shuffle=False,
                                  num_workers=16, 
                                  collate_fn=BatchCollateFn(training=False),
                                  sampler=test_sampler)
    # setup output path
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    tokens_per_image = []

    # predict and save to file
    with torch.no_grad():
        for step_idx, input_data_item in tqdm(enumerate(test_data_loader)):
            # Skip this iteration if the file was empty
            if input_data_item is None:
                continue  
            for key, input_value in input_data_item.items():
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    input_data_item[key] = input_value.to(device)
            # For easier debug.
            image_names = input_data_item["filenames"]
            # Extract text segments (tokens) for each image in the batch
            text_segments_batch = input_data_item['text_segments']  # (B, num_boxes, T)
            # Count the number of non-zero tokens for each image
            input_token_counts = [torch.sum(text_segment > 0).item() for text_segment in text_segments_batch]
            # Log the image name and the number of tokens
            # for img_name, token_count in zip(image_names, token_counts):
            #     print(f"Image: {img_name}, Tokens: {token_count}")
            output = pick_model(**input_data_item)
            logits = output['logits']                           # (B, N*T, out_dim)
            print("Logits shape:", logits.shape)
            new_mask = output['new_mask']
            print("New mask:", new_mask.shape)
            image_indexs = input_data_item['image_indexs']      # (B,)
            print("Image Indexs",image_indexs.shape)
            text_segments = input_data_item['text_segments']    # (B, num_boxes, T)
            print("Text segments:",text_segments.shape)
            mask = input_data_item['mask']
            # List[(List[int], torch.Tensor)]
            best_paths = pick_model.module.decoder.crf_layer.viterbi_tags(logits, 
                                                                          mask=new_mask, 
                                                                          logits_batch_first=True)
            predicted_tags = []
            for path, score in best_paths:
                predicted_tags.append(path)

            # convert iob index to iob
            decoded_tags_list = iob_index_to_str(predicted_tags)
            # union text as a sequence and convert index to string
            decoded_texts_list = text_index_to_str(text_segments, mask)

            for decoded_tags, decoded_texts, image_index, input_token_count in zip(decoded_tags_list, 
                                                                                    decoded_texts_list, 
                                                                                    image_indexs,
                                                                                    input_token_counts):
                # List[ Tuple[str, Tuple[int, int]] ]
                spans = bio_tags_to_spans(decoded_tags, [])
                spans = sorted(spans, key=lambda x: x[1][0])

                print("Decoded tags:", len(decoded_tags), "Span Count:", len(spans))

                entities = []  # exists one to many case
                for entity_name, range_tuple in spans:
                    entity = dict(entity_name=entity_name,
                                  text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                    entities.append(entity)
                
                # Print the input and output token sizes for each image
                image_name = Path(test_dataset.files_list[image_index]).stem
                print(f"Image: {image_name}, Input Tokens: {input_token_count}, Output Tokens: {len(entities)}")

                result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
                with result_file.open(mode='w') as f:
                    for item in entities:
                        f.write('{}\t{}\n'.format(item['entity_name'], item['text']))
    destroy_process_group()

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-GPU version of the original code')
    parser.add_argument('--checkpoint', 
                        type=str, 
                        required=True, 
                        help='path to checkpoint')
    parser.add_argument('--bt', 
                        type=str, 
                        required=True, 
                        help='path to boxes and transcripts folder')
    parser.add_argument('--impt', 
                        type=str, 
                        required=True, 
                        help='path to images folder')
    parser.add_argument('--bs', 
                        type=int, 
                        default=16, 
                        help='batch size')
    parser.add_argument('--output', 
                        type=str, 
                        default='output', 
                        help='output path')
    parser.add_argument('--local_rank', 
                        type=int, 
                        default=0, 
                        help='Local rank for distributed training')
    args = parser.parse_args()
    main(args)