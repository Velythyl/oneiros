import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from torch.utils.data import random_split

class ExpertDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def getitems(self):
        index = random.randint(0, len(self.files)-1)
        file = self.files[index]
        batch = torch.load(file).to("cpu")

        # can probably be done more quickly with torch.gather.........

        batch_size = batch.shape[0]

        ret = []
        for item in batch:
            good_data = item[:item.argmin(dim=0)[0]]
            for i in range(2):
                start_idx = random.randint(0, good_data.shape[0]-50)

                element = good_data[start_idx:start_idx+50]
                ret.append(element)
        return ret

class Loader:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_batch(self, size):
        # todo gather items from dataset, and then concat them all into a batch...

        gather = []
        while len(gather) < size:
            gather.extend(self.dataset.getitems())
        batch = torch.stack(gather, dim=0).to("cuda")[:size]

        labels = batch[:, 49, -3:]
        inputs = batch[:, :, :-3].unsqueeze(dim=1)

        return inputs, labels


def _get_dataloader(folders):

    files = []
    for folder in folders:
        files.extend(list(map(lambda x: f"{folder}/{x}", filter(lambda y: y.startswith('buffer'), os.listdir(folder)))))

    train_dataset = ExpertDataset(files)
    train_loader = Loader(train_dataset) #DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    return train_loader

def get_dataset():
    root_path = f"/home/charlie/Desktop/project-squid/discr_data/"
    cheetach_file_path = f"{root_path}train_cheetah"
    crawler_file_path = f"{root_path}train_crawler"
    squid_file_path = f"{root_path}train_squid"
    folders = [cheetach_file_path, crawler_file_path, squid_file_path]
    return _get_dataloader(folders)

if __name__ == "__main__":
    x,y = get_dataloader()


    for _ in range(100):
        x.get_batch()
        print(_)