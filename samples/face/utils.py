"""

Copyright (c) 2020 Jho Lee, (jho.lee@kakao.com)
Licensed under the MIT License (see LICENSE for details)
Written by Jho Lee

Got much inspiration from github.com/matterport/Mask_RCNN
"""

import os

import json

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image



def compare_plots(main_title="", imgs=[], titles=[]):
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(main_title)
    if imgs is not []:
        for i in range(2):
            axes[i].imshow(imgs[i])
            axes[i].set_title(titles[i])
        plt.show()
    


def open_image_as_nparray(img):
    if type(img) is str:
        img = Image.open(img)
    img_array = np.asarray(img)

    return img_array


def dig_inside(img_array, values=[]):

    rows, cols = img_array.shape

    array = img_array.copy()

    for v in values:
        for i in range(rows):
            for j in range(cols):
                if i == 0 or i == rows-1 or \
                    j == 0 or j == cols-1:
                        array[i][j] = 0
                else:
                    if img_array[i][j] == v:
                        if img_array[i][j] == img_array[i][j-1] and \
                          img_array[i][j] == img_array[i][j+1] and \
                          img_array[i][j] == img_array[i+1][j] and \
                          img_array[i][j] == img_array[i-1][j]:
                            array[i][j] = 0
    
    
    return array

def manage_the_values(img_array, values=[], mode='remain'):
    array = img_array.copy()

    rows, cols = array.shape

    if mode == 'remain':
        for i in range(rows):
            for j in range(cols):
                if array[i][j] not in values:
                    array[i][j] = 0
    elif mode == 'remove':
        for i in range(rows):
            for j in range(cols):
                if array[i][j] in values:
                    array[i][j] = 0
    
    return array

def extract_coordinates(img_array, classes=[], values=[]):
    data = {}

    rows, cols = img_array.shape

    for c, v in zip(classes, values):
        data[c] = {}
        _x = []
        _y = []
        for i in range(rows):
            for j in range(cols):
                if img_array[i][j] == v:
                    _x.append(i)
                    _y.append(j)
        data[c]['x'] = _x
        data[c]['y'] = _y
    
    return data

def load_coordiantes_from_mask(img, classes=[], values=[]):
    img_array = open_image_as_nparray(img)
    img_shell = dig_inside(img_array, values=values)
    return extract_coordinates(img_shell, classes=classes, values=values)

def load_info_from_mask(img_dir, img_id, classes=[], values=[]):
    img = os.path.join(img_dir, img_id)
    
    img_id = img_id[:-4]
    img_array = open_image_as_nparray(img)
    img_shell = dig_inside(img_array, values=values)
    height, width = img_shell.shape
    data = {}
    data[img_id] = {}
    data[img_id]['regions'] = {}
    data[img_id]['regions']['0'] = {}
    data[img_id]['regions']['0']['shape_attributes'] = extract_coordinates(img_shell, classes=classes, values=values)
    data[img_id]['regions']['0']['shape_attributes']['name'] = "polygon"
    data[img_id]['id'] = img_id
    data[img_id]['size'] = {}
    data[img_id]['size']['height'] = height
    data[img_id]['size']['width'] = width
    return data


def extend_info_into_json( data:dict, json_path:str):
    """
    """
    # Example form
    # {'342343': {
    #   'id': '342343',
    #   'regions': {
    #       'face': {
    #           '0': {
    #               'shape_atturbiutes': {
    #                   'name': "polygon",
    #                   'x': [...],
    #                   'y': [...]
    #               }
    #           }
    #       },
    #   'size': {
    #       'height': 128,
    #       'width': 64
    #    }
    #   }
    # }
    if not os.path.isfile(json_path):
        with open(json_path, 'w', encoding="utf-8") as f:
            json.dump(data, f)
            
    else:
        with open(json_path, 'r+', encoding="utf-8") as f:
            json_data = json.load(f)
            json_data.update(data)
            f.seek(0)
            json.dump(json_data, f)

    
            
