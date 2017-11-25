import torch
import torchvision as tv
import argparse
import numpy as np
import os
import re
import pandas as pd

_ORIGINAL_SIZE = 256
_ORIGINAL_SIZE = 256
_CHANNELS = 3
_ORIENTATIONS_PER_MODEL = 10

def get_model_paths(model_paths_file, screenshot_dir,filter_str=None):
 
    with open(model_paths_file, 'r') as f:
        path_suffixes = f.read().split('\n')
        if filter_str is None:
            return [os.path.join(screenshot_dir, path_suffix)
                for path_suffix in path_suffixes]
        else:
            return [os.path.join(screenshot_dir, path_suffix)
                for path_suffix in path_suffixes if filter_str in path_suffix]

def get_inputs_for_model_paths(model_paths, pattern=None):
  edge_files = []
  image_files = []
  for model_path in model_paths:
    model_name = os.path.basename(model_path)
    for orientation in range(_ORIENTATIONS_PER_MODEL):
      edges_file_path = '%s/screenshots/%s-%d_thumb_padded.bin' % (model_path, model_name, orientation)
      image_file_path = '%s/screenshots/%s-%d_thumb_padded.png' % (model_path, model_name, orientation)
      if not pattern or re.matches(image_file_path):
        edge_files.append(edges_file_path)
        image_files.append(image_file_path)
  orientations = list(range(_ORIENTATIONS_PER_MODEL))*len(model_paths)

  return edge_files, image_files, orientations

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, edges_paths,images_paths,transform=None,
                 loader=tv.datasets.folder.default_loader):
        self.edges_paths=edges_paths
        self.images_paths=images_paths
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        edge  = self.edges_paths[index]
        image = self.images_paths[index]
        img   = self.loader(image)
        edge  = np.fromfile(edge, dtype=np.float32)
        edge = edge.reshape(256,256,1)
        if self.transform is not None:
            img = self.transform(img)
            edge=self.transform(edge)
        img=2*img-1
        edge=2*edge-1
     
        return edge, img

    def __len__(self):
        n= len(self.edges_paths)
        return n

def batches():
    data_transforms_color = tv.transforms.Compose([
	    tv.transforms.ToTensor(),
	])
    paths1=get_model_paths('train_data.txt','/02958343')
    paths2=get_inputs_for_model_paths(paths1)
    edge_path=paths2[0]
    color_path=paths2[1]
    data_set=ImagesDataset(edge_path,color_path,data_transforms_color)
    train_loader = torch.utils.data.DataLoader(data_set,
		                                   batch_size=4,
		                                   shuffle=True,
		                                   num_workers=16)
    
    return train_loader



