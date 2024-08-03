import logging
import sys
import os
from pathlib import Path
import argparse
import yaml

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# Add your models here to use their libraries directly without 
# refactoring the imports in the code. Adding the libraries to 
# the beginning of the PATH.
# This line adds the 'pick' directory to the Python path
from pick.tools.inference.pickInference import PICKInference
from tools.infer.utility import check_gpu

# Create a logger object
logger = logging.getLogger(__name__)
# Set the logging level to INFO (or desired level)
logger.setLevel(logging.INFO)
# Create a handler and set its level (if needed)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(handler)

class Inference:
    def __init__(self, model_selection, **kwargs):
        
        # Extract the required parameters from kwargs or config.yml
        self.model_selection = model_selection
        self.params = kwargs
        
        #Print the keys of self.params
        logger.info(f"self.params.keys(): {self.params.keys()}")

        # Load configuration from YAML file
        with open(self.params['config_file'], 'r') as config_data:
            config_data = yaml.safe_load(config_data)
        
        # Set the attributes based on the loaded configuration
        self.checkpoint = config_data['checkpoint']
        self.bt = config_data['bt']
        self.impt = config_data['impt']
        self.bs = config_data.get('bs', 16)
        self.output = config_data.get('output', 'output')
        self.local_rank = config_data.get('local_rank', 0)
        self.num_workers = config_data.get('num_workers', 2)

        # Print the values of the keys
        # print(f"self.checkpoint: {self.checkpoint}")
        # print(f"self.bt: {self.bt}")
        # print(f"self.impt: {self.impt}")
        # print(f"self.bs: {self.bs}")
        # print(f"self.output: {self.output}")
        # print(f"self.local_rank: {self.local_rank}")

    def perform_inference(self):
        # Logic for performing inference using the selected model
        if self.model_selection == 'pytesseract_en':
            # Perform OCR inference in English
            pass

        elif self.model_selection == 'pick':
            # Perform PICK-Token Label inference
            print(f"self.checkpoint: {self.checkpoint}")
            print(f"self.bt: {self.bt}")
            print(f"self.impt: {self.impt}")
            print(f"self.bs: {self.bs}")
            print(f"self.output: {self.output}")
            print(f"self.num_workers: {self.num_workers}")

            pick_inference = PICKInference(self.checkpoint, 
                                      self.bt, 
                                      self.impt, 
                                      self.bs, 
                                      self.output,
                                      self.num_workers)
            pick_inference.spawn_processes()

        elif self.model_selection == 'deepke':
            # Perform DeepKE inference
            pass

        elif self.model_selection == 'lilt':
            # Perform LiLT-Token Label Inference
            pass

        # The following lines are left for other types of models.
        # elif {ADD_YOUR_MODEL_CONDITION} == 'YOUR_MODEL':
        #     pass

        else:
            # Handle unsupported model selection
            pass

def main():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('model_selection', type=str, help='Model selection')
    parser.add_argument('--checkpoint_path', type=str, help='Checkpoint path')
    parser.add_argument('--bt_path', type=str, help='Boxes and transcripts path')
    parser.add_argument('--impt_path', type=str, help='Images path')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--output_path', type=str, help='Output path')
    parser.add_argument('--config_file', type=str, help='Config file')
    args = parser.parse_args()

    inference = Inference(args.model_selection, **vars(args))
    inference.perform_inference()

if __name__ == '__main__':
    main()