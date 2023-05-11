import pandas as pd
from tqdm.notebook import tqdm
import os
from shutil import copyfile, move
import sys
import json

from general_json2yolo import convert_coco_json

test_path = 'test_annotation'
train_path = 'train_annotation'

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

move('train_anno.json', os.path.join(train_path, 'train_anno.json'))
move('val_anno.json', os.path.join(test_path, 'val_anno.json'))

for folder in ['labels', 'images']:
    for path in [test_path, train_path]:
        os.makedirs(os.path.join(path, folder), exist_ok=True)

convert_coco_json(train_path)
for file in tqdm(os.listdir(os.path.join('new_dir/labels/train_anno'))):
    move(os.path.join('new_dir/labels/train_anno', file), os.path.join(train_path, 'labels', file))

convert_coco_json('./test_annotation/')
for file in tqdm(os.listdir(os.path.join('new_dir/labels/val_anno'))):
    move(os.path.join('new_dir/labels/val_anno', file), os.path.join(test_path, 'labels', file))

test_labels = os.listdir(os.path.join(test_path, 'labels'))
train_labels = os.listdir(os.path.join(train_path, 'labels'))

test_labels = set(map(lambda x: x.split('.')[0], test_labels))
train_labels = set(map(lambda x: x.split('.')[0], train_labels))

images = 'rtsd-frames/rtsd-frames'
for file in os.listdir(images):
    name = file.split('.')[0]
    if name in train_labels:
        move(os.path.join(images, file), os.path.join(train_path, 'images', file))
    if name in test_labels:
        move(os.path.join(images, file), os.path.join(test_path, 'images', file))
