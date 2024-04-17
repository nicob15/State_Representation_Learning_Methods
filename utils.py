import numpy as np
import torch
from torch import nn
import cv2
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except:
    import pickle

def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(file):
    if not file[-3:] == 'pkl' and not file[-3:] == 'kle':
        file = file+'pkl'

    with open(file, 'rb') as f:
        data = pickle.load(f)

    return data

def logvar2var(log_var):
    return torch.clip(torch.exp(log_var), min=1e-5)

def add_gaussian_noise(data, noise_level=0.0, clip=True, clip_level=(0, 1)):
    if clip:
        return (data + np.random.normal(0.0, noise_level, size=data.shape)).clip(clip_level[0], clip_level[1])
    else:
        return data + np.random.normal(0.0, noise_level, size=data.shape)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def stack_frames(prev_frame, frame, size1=84, size2=84):
    prev_frame = cv2.resize(prev_frame, (size1, size2))
    frame = cv2.resize(frame, (size1, size2))
    stack_obs = np.concatenate((prev_frame, frame), axis=-1)
    return stack_obs

def add_distractor(img, size=100, is_random=False):
    distractor = np.zeros((size, size, 3))
    if is_random:
        x = np.random.randint(size, img.shape[0]-size)
        y = np.random.randint(size, img.shape[0]-size)
    else:
        x = 125
        y = 125
    offset = int(size/2)
    img[x-offset:x+offset, y-offset:y+offset, :] = distractor
    return img
