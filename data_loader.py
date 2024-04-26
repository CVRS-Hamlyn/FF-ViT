from cProfile import label
from decimal import localcontext
from functools import total_ordering
from platform import node
from random import gauss
from tkinter import N
from wsgiref.validate import InputWrapper
import torch
import torch.utils.data as data
import os
import pickle
import json
import numpy as np
from torch.utils.data import DataLoader
import math
import random
import torch.nn as nn
from PIL import Image

def load_json(path):
    f = open(path, )
    dataset = json.load(f)
    return dataset

class pCLE_Rotation_dataset(data.Dataset):
    def __init__(self,
                mode,
                k=None):
        super(pCLE_Rotation_dataset, self).__init__()
        if k is not None:
            json_path = os.path.join('./k_fold', ('{}_dataset_{}.json').format(mode, k))
        else:
            json_path = ('./{}_dataset.json').format(mode)
        self.root_path = os.path.dirname(os.getcwd())
        self.mode = mode
        self.k = k
        if mode[:5] == "train":
            self.train = True
        else:
            self.train = False
        
        
        self.data_list = load_json(json_path)
        print('{} data is loaded from {}'.format(mode, json_path))
    

    def normlization(self, data, I_max, I_min):
        if I_max == None:
            I_max = torch.max(data)
            I_min = torch.min(data)
        data_norm = (data - I_min) / (I_max - I_min)

        return data_norm

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        inputs = {}
        data_dict = self.data_list[str(idx)]
        frame_path = os.path.join(self.root_path, data_dict['frame'])
        im = torch.from_numpy(np.array(Image.open(frame_path).convert('L'), dtype=np.float32)).unsqueeze(0)
        path_elements = data_dict['frame'].split('/')
        video = torch.from_numpy(np.load(os.path.join(self.root_path, path_elements[0], path_elements[1], path_elements[2], 'video.npy')))
        inputs['frame'] = self.normlization(im, None, None)
        angle = torch.tensor(data_dict['angle'])
        if self.train == False:
            inputs['video'] = video
        inputs['angle'] = angle
        # else:
        #     if angle <= 1 and angle >= -1:
        #         inputs['angle'] = torch.tensor(0.)
        #     else:
        #         inputs['angle'] = angle
        inputs['index'] = data_dict['index']
        
        

        return inputs

class Heart_dataset(data.Dataset):
    def __init__(self):
        super(Heart_dataset, self).__init__()
        self.root_path = os.path.dirname(os.getcwd())
        
        json_path = ('./heart_dataset.json')
        self.data_list = load_json(json_path)
    

    def normlization(self, data, I_max, I_min):
        if I_max == None:
            I_max = torch.max(data)
            I_min = torch.min(data)

        data_norm = (data - I_min) / (I_max - I_min)

        return data_norm

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        inputs = {}
        data_dict = self.data_list[str(idx)]
        frame_path = os.path.join(self.root_path, data_dict['frame'])
        im = torch.from_numpy(np.array(Image.open(frame_path).convert('L'), dtype=np.float32)).unsqueeze(0)
        path_elements = data_dict['frame'].split('/')
        video = torch.from_numpy(np.load(os.path.join(self.root_path, path_elements[0], path_elements[1], path_elements[2], 'video.npy')))
        inputs['frame'] = self.normlization(im, None, None)

        inputs['video'] = video
        angle = torch.tensor(data_dict['angle'])
        inputs['angle'] = angle
        inputs['name'] = int(data_dict['id_name'])
        inputs['index'] = data_dict['index']
        return inputs