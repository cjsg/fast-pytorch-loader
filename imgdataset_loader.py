import dataflow as df
from dataflow import LMDBData, LMDBSerializer
from pickle import loads
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, IterableDataset, DataLoader
from tqdm import tqdm
from constants import IMG_SIZE, BATCH_SIZE, NUM_WORKERS, N_EPOCHS, WARMUP_STEPS, tqdm_args


DEFAULT_ROOT = os.path.expanduser('~/datasets/imgnet12/')
FILE_PATH = os.path.join(DEFAULT_ROOT, 'val/')

print(f'\nworkers={NUM_WORKERS}'
      f'\nfilepath={FILE_PATH}')

dataset = torchvision.datasets.ImageFolder(FILE_PATH)
dataset.transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()])
loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

print('Warm-up')
with tqdm(total=WARMUP_STEPS, **tqdm_args) as pbar:
    for ix, (img, lab) in enumerate(loader):
        pbar.update()
        if ix == (WARMUP_STEPS-1):
            break

print('Evaluating speed')
with tqdm(total=N_EPOCHS*len(loader), **tqdm_args) as pbar:
    for _ in range(N_EPOCHS):
        for ix, (img, lab) in enumerate(loader):
            pbar.update()
            if ix == (len(loader)-1):
                break
