import numbers

import numpy as np
import torch
from gym import ObservationWrapper
import gym


def marshall2np(item):
    if isinstance(item, numbers.Number):
        item = [item]
    if isinstance(item, list):
        item = np.array(item)
    item = item.astype(np.float64)
    return item

def augment_box(box, augment_by_box):
    low = box.low
    high = box.high

    # new_low = np.array(low.tolist() + augment_by_box.low.tolist()) # https://open.spotify.com/track/01TgrylmgyEAgsdxiqEOpL?si=43a0a7fe5e144845
    # new_high = np.array(high.tolist() + augment_by_box.high.tolist())

    
    new_low = np.hstack((low,augment_by_box.low)) # Hardcoded for now
    new_high = np.hstack((high,augment_by_box.high)) # Hardcoded for now

    # new_low = np.hstack((low,np.ones((low.shape[0],3))*augment_by_box.low)) # Hardcoded for now
    # new_high = np.hstack((high,np.ones((high.shape[0],3))*augment_by_box.high)) # Hardcoded for now

    new_box = gym.spaces.Box(low=new_low, high=new_high)
    return new_box
