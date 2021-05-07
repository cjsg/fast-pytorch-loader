import os, sys
from pickle import loads
from tqdm import tqdm

import torch
import torchvision.transforms as T

from torch.utils.data import Dataset, IterableDataset, DataLoader
from constants import IMG_SIZE, BATCH_SIZE, NUM_WORKERS, N_EPOCHS, WARMUP_STEPS, tqdm_args

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders import LMDBDataset


def list_collate_fn(batch):
    if type(batch) == list:
        return batch
    else:
        raise ValueError(f'Unknown batch type {type(batch)}')

if __name__ == '__main__':

    # SPEED TEST

    DEFAULT_ROOT = os.path.expanduser('/work/cjsimon/datasets/lmdb_imgnet12/')
    FILE_PATH = os.path.join(DEFAULT_ROOT, 'val.lmdb')

    print(f'\nworkers={NUM_WORKERS}'
          f'\nfilepath={FILE_PATH}')

    dataset = LMDBDataset(DEFAULT_ROOT, 'val', imgtype='jpeg')
    dataset.transform = T.Resize((IMG_SIZE, IMG_SIZE))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)  # , collate_fn=list_collate_fn)

    print('Warm-up')
    with tqdm(total=WARMUP_STEPS, **tqdm_args) as pbar:
        for ix, data in enumerate(loader):   # data = (img, lab)
            pbar.update()
            if ix == (WARMUP_STEPS-1):
                break

    print('Evaluating speed')
    with tqdm(total=N_EPOCHS*len(loader), **tqdm_args) as pbar:
        for _ in range(N_EPOCHS):
            for ix, data in enumerate(loader):
                pbar.update()
                if ix == (len(loader)-1):
                    break
