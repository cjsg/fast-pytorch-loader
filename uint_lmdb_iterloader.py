import dataflow as df
import cv2
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.utils.data as data
from tqdm import tqdm
from constants import IMG_SIZE, BATCH_SIZE, NUM_WORKERS, N_EPOCHS, WARMUP_STEPS, tqdm_args


DEFAULT_ROOT = os.path.expanduser('/work/cjsimon/datasets/lmdb_imgnet12_uint/')
FILE_PATH = os.path.join(DEFAULT_ROOT, 'val.lmdb')

print(f'\nworkers={NUM_WORKERS}'
      f'\nfilepath={FILE_PATH}')

class LMDBDataset(data.IterableDataset):
# from timm
    def __init__(
            self,
            root=DEFAULT_ROOT,
            split='train',
            transform=None,
    ):
        fname = 'train.lmdb' if split == 'train' else 'val.lmdb'
        root = os.path.join(root, fname)
        self.root = root
        assert os.path.exists(self.root)
        self.parser = df.LMDBSerializer.load(self.root, shuffle=(split=='train'))
        self._reset_called = False
        self.transform = transform

    def __iter__(self):
        if not self._reset_called:
            self.parser.reset_state()
            self._reset_called = True
        for img, target in self.parser:
            img = torch.tensor(img)
            target = torch.tensor(target)
            if self.transform is not None:
                img = self.transform(img)
            # if target is None:
            #     target = torch.tensor(-1, dtype=torch.long)
            yield img, target

    def __len__(self):
        if hasattr(self.parser, '__len__'):
            return len(self.parser)
        else:
            return 0


dataset = LMDBDataset(DEFAULT_ROOT, 'val')
dataset.transform = T.Resize((IMG_SIZE, IMG_SIZE))
loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

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
