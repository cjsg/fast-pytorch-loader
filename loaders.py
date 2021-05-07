import os
from pickle import loads
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

try:
    from dataflow import LMDBData  # only tensorpack.dataflow needed
except ImportError:
    from tensorpack.dataflow import LMDBData


class LMDBGetter(LMDBData):
    def __init__(self, lmdb_path, shuffle, keys=None):
        super(LMDBGetter, self).__init__(lmdb_path, shuffle, keys)
        assert self.keys is not None, "No '__keys__' entry found in lmdb file"
        self.ix_to_keys = {ix: k for (ix, k) in enumerate(self.keys)}

    def __iter__(self):  # for iterable dataset
        for k,v in super(LMDBGetter, self).__iter__():
            yield loads(v)

    def __getitem__(self, ix):  # for key-based dataset
        with self._guard:
            k = self.ix_to_keys[ix]  # k = u'{:08}'.format(ix).encode('ascii')
            v = self._txn.get(k)
            return loads(v)


class LMDBDataset(Dataset):
    def __init__(
            self,
            root,
            split='train',
            transform=None,
            transform_target=None,
            shuffle=None,
            imgtype='numpy',):
        super(LMDBDataset, self).__init__()
        fname = 'train.lmdb' if split == 'train' else 'val.lmdb'
        lmdb_path = os.path.join(root, fname)
        self.lmdb_path = lmdb_path
        assert os.path.exists(self.lmdb_path)

        assert imgtype in ['numpy', 'jpeg']
        self.imgtype = imgtype

        shuffle = (split=='train') if shuffle is None else shuffle

        self.getter = LMDBGetter(self.lmdb_path, shuffle=shuffle)

        self._called_reset_state = False
        self.transform = transform
        self.transform_target = transform_target

    def __getitem__(self, ix):
        if not self._called_reset_state:
            self.getter.reset_state()
            self._called_reset_state = True
        img, target = self.getter.__getitem__(ix)
        if self.imgtype == 'numpy':
            img = torch.tensor(img)
        elif self.imgtype == 'jpeg':
            # img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = np.asarray(Image.open(BytesIO(img)).convert('RGB'))
            img = torch.tensor(img).permute(2,0,1).contiguous()
        else:
            ValueError('imgtype must be jpeg or numpy')
        target = torch.tensor(target)
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.tensor(-1, dtype=torch.long)
        if self.transform_target is not None:
            target = self.transform_target
        return img, target

    def __len__(self):
        if hasattr(self.getter, '__len__'):
            return len(self.getter)
        else:
            return 0


class LMDBIterDataset(IterableDataset):
    def __init__(
            self,
            root,
            split='train',
            transform=None,
            transform_target=None,
            shuffle=None,
            imgtype='numpy',  # or 'jpeg'
    ):
    # based on timm's ImageDataset
        super(LMDBIterDataset, self).__init__()
        fname = 'train.lmdb' if split == 'train' else 'val.lmdb'
        lmdb_path = os.path.join(root, fname)
        self.lmdb_path = lmdb_path
        assert os.path.exists(self.lmdb_path)

        assert imgtype in ['numpy', 'jpeg']
        self.imgtype = imgtype

        shuffle = (split=='train') if shuffle is None else shuffle
        self.getter = LMDBGetter(self.lmdb_path, shuffle=shuffle)
        # Alternatively: LMDBSerializer.load(self.root, shuffle=shuffle)
        self._called_reset_state = False
        self.transform = transform
        self.transform_target = transform_target


    def __iter__(self):
        if not self._called_reset_state:
            self.getter.reset_state()
            self._reset_called = True
        for img, target in self.getter:
            if self.imgtype == 'numpy':
                img = torch.tensor(img)
            elif self.imgtype == 'jpeg':
                # img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = np.asarray(Image.open(BytesIO(img)).convert('RGB'))
                img = torch.tensor(img).permute(2,0,1).contiguous()
            else:
                ValueError('imgtype must be jpeg or numpy')
            target = torch.tensor(target)
            if self.transform is not None:
                img = self.transform(img)
            if target is None:
                target = torch.tensor(-1, dtype=torch.long)
            if self.transform_target is not None:
                target = self.transform_target(target)
            yield img, target

    def __len__(self):
        if hasattr(self.getter, '__len__'):
            return len(self.getter)
        else:
            return 0