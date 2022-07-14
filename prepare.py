# prepare the dataset to train image-mask-flow model
import os
import json
from PIL import Image
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as vf
from torchvision import utils as vutils
from scipy.io import loadmat

DATA_ROOT = 'dataset/river_fukui'
DATA_TYPES = ['image', 'mask', 'flow']
IM_SIZE = 512

image = os.path.join(DATA_ROOT, DATA_TYPES[0], 'frame_0000046.jpg')
mask = os.path.join(DATA_ROOT, DATA_TYPES[1], 'frame_0000046.png')
flow = os.path.join(DATA_ROOT, DATA_TYPES[2], 'frame_0000046.mat')

transform_list1 = [
    transforms.Resize((int(IM_SIZE),int(IM_SIZE))),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]
trans = transforms.Compose(transform_list1)


def transform_image(fpath):
    raw_image = Image.open(fpath)
    return trans(raw_image)


def transform_mask(fpath, im_size=IM_SIZE):
    raw_image = Image.open(fpath)
    resized_image = vf.resize(raw_image, (int(im_size),int(im_size)))
    resized_image_t = torch.from_numpy(np.array(resized_image)).float()
    return resized_image_t
 

def transform_flow(fpath, im_size=IM_SIZE):
    raw_ndarray = loadmat(fpath)['flow']
    resized_ndarray = cv2.resize(raw_ndarray, dsize=(im_size, im_size), interpolation=cv2.INTER_CUBIC)
    resized_image_t = torch.from_numpy(resized_ndarray)
    return resized_image_t.permute(2,0,1)


def get_all_file_id():
    # for file 'frame_[id].jpg', returns the 'id'
    names = sorted(os.listdir(os.path.join(DATA_ROOT, DATA_TYPES[0])))
    return [n[:-4].split('_')[-1] for n in names]


def build_order_dict(ids):
    # {curr: next} if no next then None
    dict = OrderedDict()
    none_count = 0
    for i in ids:
        digits = len(i)
        next_id = f'{int(i) + 1:0>{digits}}'
        if next_id in ids:
            dict[i] = next_id
        else:
            dict[i] = None
            none_count += 1
    return dict, none_count


def save_as_instance_json(dict):
    instance_json = {}
    samples = []
    
    prefix = 'frame_'
    s_id = 0
    for k in dict:
        next_id = dict[k]
        if next_id is not None:  # filter out those final frames 
            instance = {}
            instance['id'] = s_id
            instance['image'] = os.path.join(DATA_ROOT, DATA_TYPES[0], f'{prefix}{k}.jpg')
            instance['mask'] = os.path.join(DATA_ROOT, DATA_TYPES[1], f'{prefix}{k}.png')
            instance['flow'] = os.path.join(DATA_ROOT, DATA_TYPES[2], f'{prefix}{k}.mat')
            instance['next_image'] = os.path.join(DATA_ROOT, DATA_TYPES[0], f'{prefix}{next_id}.jpg')
            instance['next_mask'] = os.path.join(DATA_ROOT, DATA_TYPES[1], f'{prefix}{next_id}.png')
            instance['next_flow'] = os.path.join(DATA_ROOT, DATA_TYPES[2], f'{prefix}{next_id}.mat')
            samples.append(instance)
            s_id += 1
    instance_json['instances'] = samples

    with open(os.path.join(DATA_ROOT, 'instances_fukui.json'), 'w') as f:
        json.dump(instance_json, f)    
    return instance_json




