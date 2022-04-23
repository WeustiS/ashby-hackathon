import torch
import numpy as np
from netCDF4 import Dataset
import cartopy

class AshbyDataset(torch.utils.data.Dataset):
    def __init__(self, path, train=True, transforms=None):
        
        self.path = path
        self.train = train
        self.transforms = (lambda x: x) if transforms is None else transforms
        
        
    def __len__(self):
        pass # TODO
    
    def __getitem__(self, idx):
        pass # TODO