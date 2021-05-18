import os
from pickle import loads
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import random
from torch.utils.data import Dataset, IterableDataset, get_worker_info, DataLoader
from torch.utils.data.dataloader import default_collate

try:
    from dataflow import LMDBData  # only tensorpack.dataflow needed
except ImportError:
    from tensorpack.dataflow import LMDBData


class LMDBGetter(LMDBData):
    def __init__(self, lmdb_path, shuffle, keys=None):
        super(LMDBGetter, self).__init__(lmdb_path, shuffle, keys)
        assert self.keys is not None, "No '__keys__' entry found in lmdb file"
        self.ix_to_keys = {ix: k for (ix, k) in enumerate(self.keys)}
        self._start = 0
        self._end = len(self)

    def __iter__(self):  # for iterable dataset
        # for k,v in super(LMDBGetter, self).__iter__():
        #     yield loads(v)
        with self._guard:
            for ix in range(self._start, self._end):
                k = self.ix_to_keys[ix]
                v = self._txn.get(k)
                yield loads(v)

    def __getitem__(self, ix):  # for map-style dataset
        with self._guard:
            k = self.ix_to_keys[ix]  # k = u'{:08}'.format(ix).encode('ascii')
            v = self._txn.get(k)
            return loads(v)

    @property
    def len_per_worker(self):
        return self._end - self._start


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
        assert os.path.exists(self.lmdb_path), f'Non existing path {self.lmdb_path}'

        assert imgtype in ['numpy', 'jpeg'], f'Unknonwn imgtype {imgtype}'
        self.imgtype = imgtype

        shuffle = (split=='train') if shuffle is None else shuffle

        self.getter = LMDBGetter(self.lmdb_path, shuffle=shuffle)

        self._initialized_worker = False
        self.transform = transform
        self.transform_target = transform_target

    def __getitem__(self, ix):
        if not self._initialized_worker:
            self.getter.reset_state()
            self._initialized_worker = True
        # img, target = self.getter.__getitem__(ix)
        img, target = self.getter[ix]
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
            imgtype='numpy',  # or 'jpeg'
    ):
    # based on timm's ImageDataset
        super(LMDBIterDataset, self).__init__()
        fname = 'train.lmdb' if split == 'train' else 'val.lmdb'
        lmdb_path = os.path.join(root, fname)
        self.lmdb_path = lmdb_path
        assert os.path.exists(self.lmdb_path), f'Non existing path {self.lmdb_path}'

        assert imgtype in ['numpy', 'jpeg'], f'Unknonwn imgtype {imgtype}'
        self.imgtype = imgtype

        self.getter = LMDBGetter(self.lmdb_path, shuffle=False)
        # Alternatively: LMDBSerializer.load(self.root, shuffle=False)
        self.transform = transform
        self.transform_target = transform_target

        self._initialized_worker = False

    def initialize_worker(self):
        '''
            When loading this iterative dataset with multiple workers, then
            each worker gets a different chunk of the dataset to load, and
            loads it sequentially.
        '''
        self.getter.reset_state()
        worker_info = get_worker_info()
        if worker_info is not None:  # multi-process loading: we are in a worker
            per_worker = int(np.ceil((self.getter._end - self.getter._start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            self.getter._start = self.getter._start + worker_id * per_worker
            self.getter._end = min(self.getter._start + per_worker, self.getter._end)

    def __iter__(self):
        if not self._initialized_worker:
            self.initialize_worker()
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
        return self.getter._end - self.getter._start
        # if hasattr(self.getter, '__len__'):
        #     return len(self.getter)
        # else:
        #     return 0


class BufferedDataset(IterableDataset):
    def __init__(self, iterable_dataset, buffer_size, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
           *, prefetch_factor=2, persistent_workers=False):

        super(BufferedDataset, self).__init__()

        # multiply prefetch_factor by batch_size, because we set batch_size to 1
        self.loader = DataLoader(
            iterable_dataset, batch_size=None, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, timeout=timeout,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor,  # x batch-size
            persistent_workers=persistent_workers)

        self.buffer_size = buffer_size
        self.buf = []

    def __len__(self):
        return len(self.loader.dataset)

    def __iter__(self):
        for x in self.loader:
            if len(self.buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield self.buf[idx]
                self.buf[idx] = x
            else:
                self.buf.append(x)
        random.shuffle(self.buf)
        while self.buf:
            yield self.buf.pop()


class BufferedDataLoader(object):
    def __init__(self, buffer_size, dataset, batch_size, drop_last=False, **loader_kwargs):
        # loader_args and loader_kwargs should not contain the dataset,
        # batch_size, drop_last and collate_fn argument
        loader_collate_fn = lambda data_list: data_list
        if 'collate_fn' in loader_kwargs:
            self.collate_fn = loader_kwargs['collate_fn']
        else:
            self.collate_fn = default_collate
        loader_kwargs['collate_fn'] = loader_collate_fn

        if 'generator' in loader_kwargs:
            self.generator = loader_kwargs['generator']
        else:
            self.generator = torch.Generator()

        assert buffer_size >= batch_size, 'buffer_size must be >= batch_size'
        # NB: drop_last is always set to False in the internal data-loader. The
        # `drop_last`argument is only used for the BufferedDataLoader
        self.loader = DataLoader(
            dataset, batch_size, drop_last=False, **loader_kwargs)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.buffer_size = buffer_size
        self.buffer = []

    def __iter__(self):
        batch, ixs = [], []
        for data_batch in self.loader:
            ixs = torch.randint(
                high=self.buffer_size, size=(len(data_batch),), dtype=torch.int64,
                generator=self.generator).tolist()
            for data in data_batch:
                if len(self.buffer) == self.buffer_size:
                    ix = ixs.pop()
                    batch.append(self.buffer[ix])
                    self.buffer[ix] = data
                    if len(batch) == self.batch_size:
                        yield self.collate_fn(batch)
                        batch = []
                else:
                    self.buffer.append(data)
        random.shuffle(self.buffer)
        while self.buffer:
            batch.append(self.buffer.pop())
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield self.collate_fn(batch)

    # def __iter__(self):
    #     batch = []
    #     for data_batch in self.loader:
    #         if len(self.buffer) >= self.buffer_size:
    #             # sample a batch
    #             ixs = torch.randint(
    #                 high=len(self.buffer), size=(len(data_batch),), dtype=torch.int64,
    #                 generator=torch.Generator()).tolist()
    #             for ix in ixs:
    #                 batch.append(self.buffer[ix])
    #                 self.buffer[ix] = data_batch.pop()
    #             if len(batch) == self.batch_size:
    #                 yield self.collate_fn(batch)
    #                 batch = []
    #         else:
    #             self.buffer.extend(data_batch)
    #     random.shuffle(self.buffer)
    #     while self.buffer:
    #         batch.append(self.buffer.pop())
    #         if len(batch) == self.batch_size:
    #             yield batch
    #             batch = []
    #     if len(batch) > 0 and 

    def __len__(self):
        return len(self.loader)

# class BufferedDataLoader(DataLoader):
#     def __init__(self, buf_size, dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None,
#            pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
#            *, prefetch_factor=2, persistent_workers=False):
# 
#         super(BufferedDataLoader, self).__init__(
#             dataset, batch_size=batch_size, shuffle=shuffle,
#             num_workers=num_workers, collate_fn=collate_fn,
#             pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
#             worker_init_fn=worker_init_fn, *, prefetch_factor=prefetch_factor,
#             persistent_workers=persistent_workers)
# 
#         self.loader = DataLoader(
#             dataset, batch_size=None, shuffle=False, num_workers=num_workers,
#             pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
#             worker_init_fn=worker_init_fn, *,
#             prefetch_factor=prefetch_factor*batch_size,  # multiply by batch_size, because we set batch_size to 1
#             persistent_workers=persistent_workers)
# 
#         self.batch_size = batch_size
#         self.sampler = 
#         self.batch_sampler = 
#         self.collate_fn = default_collate
#         self.buf_size = buf_size
#         self.buffer = []
# 
#     def __iter__(self):
#         for data in self.loader:
#             if len(self.buffer) == buf_size:
                
