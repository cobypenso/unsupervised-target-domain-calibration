#!/usr/bin/env python
#coding:utf-8
from os import walk
import os
from sklearn.model_selection import train_test_split
from collections import Counter

office_labels = {'tape_dispenser': 0, 'bike_helmet': 1, 
                 'paper_notebook': 2, 'stapler': 3, 'calculator': 4, 'printer': 5, 'back_pack': 6, 
                 'desk_chair': 7, 'desktop_computer': 8, 'laptop_computer': 9,  'bike': 10, 'bookcase': 11, 
                 'phone': 12, 'punchers': 13, 'pen': 14, 
                 'projector': 15, 'ring_binder': 16, 'ruler': 17, 'headphones': 18, 
                 'letter_tray': 19, 'bottle': 20, 'scissors': 21, 'desk_lamp': 22, 'mouse': 23, 'trash_can': 24, 
                 'monitor': 25, 'speaker': 26, 'file_cabinet': 27, 'keyboard': 28, 'mug': 29, 'mobile_phone': 30}

def split_data(file_name):
    
    source_list = open(file_name).readlines()
    labels = list(map(lambda x: int(x.split(' ')[-1]), source_list))
    source_train, source_val = train_test_split(source_list, test_size=0.2, stratify = labels)
    print(len(source_train))
    print(len(source_val))

    # 
    train_labels = list(map(lambda x: int(x.split(' ')[-1]), source_train))
    val_labels = list(map(lambda x: int(x.split(' ')[-1]), source_val))
    
    train_labels.sort()
    val_labels.sort()

    train_labels_counter = dict(Counter(train_labels))
    val_labels_counter = dict(Counter(val_labels))

    train_labels_counter = {k:train_labels_counter[k] for k in sorted(train_labels_counter)}
    val_labels_counter = {k:val_labels_counter[k] for k in sorted(val_labels_counter)}


    print (train_labels_counter)
    print (val_labels_counter)

    source_train_file_name = file_name.replace('list', 'train_list')
    source_val_file_name = file_name.replace('list', 'val_list')

    source_train_file = open(source_train_file_name, "w")
    for line in source_train:
        source_train_file.write(line)

    source_val_file = open(source_val_file_name, "w")
    for line in source_val:
        source_val_file.write(line)


def create_txt_file(path_name):
    domain_name = path_name.split('/')[-2]
    print (domain_name)

    with open(f'office31/{domain_name}_list.txt', 'a') as f:

        all_images = []
        print (path_name)
        fileList = os.listdir(path_name)
        for file in fileList:
            folder = os.path.join(path_name, file)
            images = os.listdir(folder)
            for image in images:
                image = os.path.join(folder, image)

                line = image + ' ' + str(office_labels[file])
                line = line.replace('../pytorch/data', './data')

                all_images.append(line)

                f.write(line + "\n")


# create_txt_file('../pytorch/data/office31/amazon/images')
# create_txt_file('../pytorch/data/office31/webcam/images')
# create_txt_file('../pytorch/data/office31/dslr/images')

file_name_list = ['office31/amazon_list.txt', 'office31/webcam_list.txt', 'office31/dslr_list.txt']
# 'office-home/Art_list.txt', 'office-home/Clipart_list.txt', 'office-home/Product_list.txt','office-home/Real_World_list.txt',
# 'image-clef/b_list.txt', 'image-clef/c_list.txt', 'image-clef/i_list.txt','image-clef/p_list.txt',
# ]
# file_name_list = ['office-home/Art_list.txt', 'office-home/Clipart_list.txt', 'office-home/Product_list.txt','office-home/Real_World_list.txt']

for file_name in file_name_list:
    split_data(file_name)
