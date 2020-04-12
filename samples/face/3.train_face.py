#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
from IPython import get_ipython

from keras import backend as K

import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project


ROOT_DIR = os.path.abspath("../..")


# In[2]:


# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from samples.face import face



# In[3]:


# Directory to save trained models
MODEL_DIR = os.path.join(ROOT_DIR, "logs/weights")


# ## Notebook Preferences

# In[4]:


def get_ax(rows=1, cols=1, size=8):
    """
    Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default isze attribute to control the size
    of rendered images
    """
    
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Configurations
# 
# Configurations are defined in face.py
# 

# In[5]:


config = face.FaceConfig()
config.display()

config.IMAGE_MAX_DIM = 512 # Override the resizing options from 256 to 1024.
config.STEPS_PER_EPOCH = 3200 # Override the value of steps per epoch


FACE_DIR = os.path.join(ROOT_DIR, "samples/face/face_data")

# Directory to save weights
FACE_MODEL_DIR = os.path.join(MODEL_DIR, 'face')

# Which weights to start with?
init_weight = "coco"
custom_weight_path = os.path.join(FACE_MODEL_DIR, "coco/face_epochs10(5)_steps3200_resize512")

# Set epochs
head_epochs = 2
all_epochs = 4

tag = "coco_epochs2h-2a"

# Directory to save events
import datetime

EVENT_DIR = os.path.join(ROOT_DIR, "logs/events/face_{}_{:%Y%m%dT%H%M}".format(
    tag, datetime.datetime.now()))


# Print this jupyter file's configurations


# ## Dataset

# In[6]:


# Load dataset
# Get the dataset 'CelebA'

# dataset = face.FaceDataset()
# dataset.load_face(FACE_DIR, "train")

# Must call before using the dataset
# dataset.prepare()

# print("Image Count: {}".format(len(dataset.image_ids)))
# print("Class Count: {}".format(dataset.num_classes))
# for i, info in enumerate(dataset.class_info):
#     print("{:3}. {:50}".format(i, info['name']))


# ### Training dataset

# In[7]:


# Training dataset
dataset_train = face.FaceDataset()
dataset_train.load_face(FACE_DIR, 'train')
dataset_train.prepare()

print("Image Count: {}".format(len(dataset_train.image_ids)))
print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# ### Validation Dataset

# In[8]:


# Validation dataset
dataset_val = face.FaceDataset()
dataset_val.load_face(FACE_DIR, 'val')
dataset_val.prepare()

print("Image Count: {}".format(len(dataset_val.image_ids)))
print("Class Count: {}".format(dataset_val.num_classes))
for i, info in enumerate(dataset_val.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# In[9]:


# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# ## Create Model

# In[10]:


# Create model in training mode
model = modellib.MaskRCNN(
        mode="training", 
        config=config,
        model_dir=MODEL_DIR)


# In[11]:



# Directory to save logs and trained model

if init_weight == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_weight == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
    # Directory to save logs and trained model
    
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                 "mrcnn_bbox", "mrcnn_mask"])
    
elif init_weight == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
elif init_weight == "custom":
    if not os.path.exists(custom_weight_path):
        raise FileNotFoundError
    model.load_weights(custom_weight_path)


# ## Training
# 
# Train in two stages:
# 
#  1. Only the heads. Here we're freezing all the backbone layers and training only the randomly intialized layers
#  (.e. the ones that we didn't use pre-trained weights from MS COCO).
#  To train only the head layers, pass layers='heads' to the train() function.
#  2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process.
#  Simply pass layers="all to train all layers.
#  
# 

# In[ ]:


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

model.train(dataset_train, dataset_val,
            event_dir=EVENT_DIR,
            learning_rate=config.LEARNING_RATE,
            epochs=head_epochs,
            layers='heads')


# In[ ]:


# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            event_dir=EVENT_DIR,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=all_epochs,
            layers="all")


# In[ ]:


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.200324.h5")
# model.keras_model.save_weights(model_path)


import pathlib
pathlib.Path(FACE_MODEL_DIR).mkdir(exist_ok=True)

model_path = os.path.join(FACE_MODEL_DIR, init_weight)

model_path = os.path.join(model_path, 'face_epochs{}({})_steps{}_resize{}.h5'.format(all_epochs, head_epochs, config.STEPS_PER_EPOCH, config.IMAGE_MAX_DIM))
model.keras_model.save_weights(model_path)

print("weights saved to {}".format(model_path))


# In[ ]:




