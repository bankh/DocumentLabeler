## Data format example
This implementation requires the input data in the following format:
### Training/Validation data
1. `train_samples_list.csv` file: multi-line with format: `index,document_type,file_name`.
2. `boxes_and_transcripts` folder: `file_name.tsv` files.
    * every `file_name.tsv` file has multi-line with format: `index,box_coordinates (clockwise 8 values),
    transcripts,box_entity_types` .
3. `images` folder:  `file_name.jpg` files.
4. `entities` folder (optional) : `file_name.txt` files.
    * every `file_name.txt` file contains a json format string, providing the exactly label value of
    every entity.
    * if `iob_tagging_type` is set to `box_level`, this folder will not be used, then `box_entity_types` in
     `file_name.tsv` file of `boxes_and_transcripts` folder will be used as label of entity.
      otherwise, it must be provided.
### Testing data
1. `boxes_and_transcripts` folder: `file_name.tsv` files
    * every `file_name.tsv` file has multi-line with format: `index,box_coordinates (clockwise 8 values),
    transcripts`.
2. `images` folder:  `file_name.jpg` files.

## How to use the shell scripts for data preparation

Here, label_list.txt indicates the labels that we have for the specific dataset-of-interest. This text files are under the root folder of the dataset. For example, SROIE's label_list.txt is under ./SROIE and similarly DocBank's label_list.txt is under ./DocBank/.

### To extract the label structure for the dataset:

evaluate_labels.sh  

The sample below is for training and testing part of 100 pages
```
$ ./evaluate_labels.sh ./dataset_100/train/boxes_and_transcripts/ label_list.txt > ./dataset_100/train/label_structure.txt
$ ./evaluate_labels.sh ./dataset_100/test/boxes_and_transcripts/ label_list.txt > ./dataset_100/test/label_structure.txt
```
The sample below is for training and testing part of 1000 pages
```
$ ./evaluate_labels.sh ./dataset_1000/train/boxes_and_transcripts/ label_list.txt > ./dataset_1000/train/label_structure.txt
$ ./evaluate_labels.sh ./dataset_1000/test/boxes_and_transcripts/ label_list.txt > ./dataset_1000/test/label_structure.txt
```
The sample below is for training and testing part of 10000 pages
```
$ ./evaluate_labels.sh ./dataset_10000/train/boxes_and_transcripts/ label_list.txt > ./dataset_10000/train/label_structure.txt
$ ./evaluate_labels.sh ./dataset_10000/test/boxes_and_transcripts/ label_list.txt > ./dataset_10000/test/label_structure.txt
```

### To copy the files in order based on the filenames in the boxes_and transcripts while preserving the folder structure intact  

copy_folders.sh  

The sample below is for training and testing part of 100 pages
```
$ ./copy_folders.sh ./DocBank_all/train/ ./dataset_100/train/ 100
$ ./copy_folders.sh ./DocBank_all/test/ ./dataset_100/test/ 100
```

The sample below is for training and testing part of 1000 pages
```
$ ./copy_folders.sh ./DocBank_all/train/ ./dataset_1000/train/ 1000
$ ./copy_folders.sh ./DocBank_all/test/ ./dataset_1000/test/ 1000
```

The sample below is for training and testing part of 10000 pages
```
$ ./copy_folders.sh ./DocBank_all/train/ ./dataset_10000/train/ 10000
$ ./copy_folders.sh ./DocBank_all/test/ ./dataset_10000/test/ 10000
```

### To generate csv files that required by PICK framework in each train and test folder (works as an index)  

csv_generate.sh  

The sample below is for training and testing part of 100 pages
```
$ ./csv_generate.sh /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/DocBank_all/train/train_samples_list.csv /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_100/train/train_samples_list.csv ./dataset_100/train/boxes_and_transcripts/

$ ./csv_generate.sh /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/DocBank_all/test/test_samples_list.csv /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_100/test/test_samples_list.csv ./dataset_100/test/boxes_and_transcripts/
```

The sample below is for training and testing part of 1000 pages
```
$ ./csv_generate.sh /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/DocBank_all/train/train_samples_list.csv /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_1000/train/train_samples_list.csv ./dataset_1000/train/boxes_and_transcripts/

$ ./csv_generate.sh /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/DocBank_all/test/test_samples_list.csv /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_1000/test/test_samples_list.csv ./dataset_1000/test/boxes_and_transcripts/
```

The sample below is for training and testing part of 10000 pages
```
$ ./csv_generate.sh /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/DocBank_all/train/train_samples_list.csv /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000/train/train_samples_list.csv ./dataset_10000/train/boxes_and_transcripts/

$ ./csv_generate.sh /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/DocBank_all/test/test_samples_list.csv /mnt/data_drive/CSU_PhD/research/software/PICK-pytorch/data/DocBank/dataset_10000/test/test_samples_list.csv ./dataset_10000/test/boxes_and_transcripts/
```