import os
from pickle import loads
from tqdm import tqdm

from dataflow import LMDBData, LMDBSerializer
import dataflow as df
import torch
import torchvision.transforms as T

from torch.utils.data import Dataset, IterableDataset, DataLoader
from constants import IMG_SIZE, BATCH_SIZE, NUM_WORKERS, N_EPOCHS, WARMUP_STEPS, tqdm_args


DEFAULT_ROOT = os.path.expanduser('/work/cjsimon/datasets/lmdb_imgnet12_uint/')
FILE_PATH = os.path.join(DEFAULT_ROOT, 'train.lmdb')

print(f'\nworkers={NUM_WORKERS}'
      f'\nfilepath={FILE_PATH}')

class LMDBGetter(LMDBData):
    def __init__(self, lmdb_path, shuffle, keys=None):
        super(LMDBGetter, self).__init__(lmdb_path, shuffle, keys)
        assert self.keys is not None, "No '__keys__' entry found in lmdb file"
        self.ix_to_keys = {ix: k for (ix, k) in enumerate(self.keys)}

    def __getitem__(self, ix):
        with self._guard:
            k = self.ix_to_keys[ix]  # k = u'{:08}'.format(ix).encode('ascii')
            v = self._txn.get(k)
            return loads(v)


class LMDBDataset(Dataset):
    def __init__(
            self,
            root=DEFAULT_ROOT,
            split='train',
            transform=None,
            shuffle=None
    ):
    # based on timm's ImageDataset
        super(LMDBDataset, self).__init__()
        fname = 'train.lmdb' if split == 'train' else 'val.lmdb'
        lmdb_path = os.path.join(root, fname)
        self.lmdb_path = lmdb_path
        assert os.path.exists(self.lmdb_path)

        shuffle = (split=='train') if shuffle is None else shuffle
        self.getter = LMDBGetter(self.lmdb_path, shuffle=shuffle)
        self._called_reset_state = False
        self.transform = transform

    def __getitem__(self, ix):
        if not self._called_reset_state:
            self.getter.reset_state()
            self._called_reset_state = True
        img, target = self.getter.__getitem__(ix)
        img = torch.tensor(img)
        target = torch.tensor(target)
        if self.transform is not None:
            img = self.transform(img)
        return img, target  # use yield?

    def __len__(self):
        if hasattr(self.getter, '__len__'):
            return len(self.getter)
        else:
            return 0


dataset = LMDBDataset(DEFAULT_ROOT, 'val')
dataset.transform = T.Resize((IMG_SIZE, IMG_SIZE))
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
            img = img.cuda()
            lab = lab.cuda()
            pbar.update()
            if ix == (len(loader)-1):
                break
