# -*- coding: utf-8 -*-

## Script for preprocessing dataset
import os
import pandas
import json
import csv
import shutil

## Input dataset
data_path = "./dataset/ICDAR-2019-SROIE/data/"
box_path = data_path + "box/"
img_path = data_path + "img/"
key_path = data_path + "key/"

## Output dataset
out_boxes_and_transcripts = "./boxes_and_transcripts/"
out_images = "./images/"
out_entities = "./entities/"

train_samples_list = []
for file in os.listdir(data_path + "box/"):

    # Reading CSV
    with open(box_path + file, "r") as fp:
        reader = csv.reader(fp, delimiter=",")
        ## Arranging dataframe index, coordinates, x1_1, y1_1, x2_1, y2_1, x3_1, y3_1, x4_1, y4_1, transcript
        rows = [[1] + x[:8] + [','.join[8:]).strip(',')] for x in reader]
        df = pandas.DataFrame(rows)