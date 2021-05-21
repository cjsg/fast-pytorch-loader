import os
from pickle import loads
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import random
from torch.utils.data import Dataset, IterableDataset, get_worker_info, DataLoader
from torch.utils.data.dataloader import default_collate
import warnings
from time import sleep
from torch.multiprocessing import Lock

try:
    from dataflow import LMDBData  # only tensorpack.dataflow needed
except ImportError:
    from tensorpack.dataflow import LMDBData


class LMDBGetter(LMDBData):
    '''
        This class is used internally both by the LMDBDataset class (which uses
        `__get_item__`) and the LMDBIterDataset (which uses `__iter__`)
    '''
    def __init__(self, lmdb_path, shuffle, keys=None):
        super(LMDBGetter, self).__init__(lmdb_path, shuffle, keys)
        assert self.keys is not None, "No '__keys__' entry found in lmdb file"
        self.ix_to_keys = {ix: k for (ix, k) in enumerate(self.keys)}
        self._start = self._worker_start = 0
        self._end = self._worker_end = len(self)

    def __iter__(self):  # for iterable dataset
        # for k,v in super(LMDBGetter, self).__iter__():
        #     yield loads(v)
        with self._guard:
            for ix in range(self._worker_start, self._worker_end):
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
        return self._worker_end - self._worker_start


class LMDBDataset(Dataset):
    '''
        A map-style dataset that opens an LMDB file and randomly loads its
        files/images using the LMDB's dictionnary/key-value based access.
        Note that the LMDB file is opened only once.
    '''

    def __init__(self, root, split='train', transform=None,
                 transform_target=None, shuffle=None, imgtype='numpy',):
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
            # np.asarray does not copy the underlying data. But this PyTorch
            # throws a long warning, because PIL.Image apparently returns
            # non-writable arrays. Anyway, using np.array has no noticeable
            # performance decrease.
            # img = np.asarray(Image.open(BytesIO(img)).convert('RGB'))
            img = np.array(Image.open(BytesIO(img)).convert('RGB'))
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
    '''
        An iterable dataset that opens an LMDB file and reads it in the order
        of its keys (which should be the order of storage). If multiple workers
        are used, each worker gets a different chunk of the dataset.

        Argument split should be 'train' or 'val'.
    '''

    def __init__(self, root, split='train', transform=None,
                 transform_target=None, imgtype='numpy',):  # or 'jpeg'):
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
        self._initialized_worker = True
        self.getter.reset_state()
        worker_info = get_worker_info()
        if worker_info is not None:  # multi-process loading: we are in a worker
            per_worker = int(np.ceil((self.getter._end - self.getter._start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            self.getter._worker_start = self.getter._start + worker_id * per_worker
            self.getter._worker_end = min(self.getter._worker_start + per_worker, self.getter._end)

    def __iter__(self):
        if not self._initialized_worker:
            self.initialize_worker()
        for img, target in self.getter:
            if self.imgtype == 'numpy':
                img = torch.tensor(img)
            elif self.imgtype == 'jpeg':
                # np.asarray does not copy the underlying data. But this
                # PyTorch throws a long warning, because PIL.Image apparently
                # returns non-writable arrays. Anyway, using np.array has no
                # noticeable performance decrease.
                # img = np.asarray(Image.open(BytesIO(img)).convert('RGB'))
                img = np.array(Image.open(BytesIO(img)).convert('RGB'))
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

    def worker_len(self):
        return self.getter._worker_end - self.getter._worker_start


class SafeBuffer(object):
    '''
        This class provides a thread-safe buffer to be used by the
        `BufferedDataLoader`, which will pass this buffer to every of its
        workers (the dataset-loading sub-processes).
    '''

    def __init__(self, buffer_size: int, num_workers: int):
        self._list = []
        self._lock = Lock()
        self._buffer_size = buffer_size
        self._num_workers = num_workers
        self._num_workers_done_looping = 0
        self._ixs = []

    def append(self, x):
        self._lock.acquire()
        self._list.append(x)
        self._lock.release()

    def pop(self):
        self._lock.aquire()
        x = self._list.pop()
        self._lock.release()
        return x

    def random_get_and_replace(self, x):
        self._lock.acquire()
        ix = self.sample_random_ix()
        out = self._list[ix]
        self._list[ix] = x
        self._lock.release()
        return out

    def sample_random_ix(self):
        if len(self._ixs) == 0:
            self._ixs = torch.randint(high=len(self), size=(100,),
                                      dtype=torch.int64).tolist()
        return self._ixs.pop()

    @property
    def all_workers_done(self):
        return (self._num_workers_done_looping >= self._num_workers)

    def send_done_looping(self):
        self._lock.acquire()
        self._num_workers_done_looping += 1
        if self._num_workers_done_looping == self._num_workers:
            random.shuffle(self._list)
        self._lock.release()

    def reset(self):
        self._num_workers_done_looping = 0

    def __len__(self):
        return len(self._list)


class BufferedDataset(IterableDataset):
    '''
        This class is a wrapper for iterable datasets that is used by the
        BufferedDataLoader to handle access to a joint buffer between
        different workers.
    '''
    def __init__(self, dataset):
        super(BufferedDataset, self).__init__()
        self.dataset = dataset
        self._initialized_buffer = False
        self._iterdataset = None

    def initialize_buffer(self, buffered_dataloader): # buf, buffer_size, persistent_buffer):
        # The buffer comes from the loader and is the same for all workers
        self._buffer = buffered_dataloader.buffer
        self._buffer_size = buffered_dataloader.buffer_size
        self._persistent_buffer = buffered_dataloader.persistent_buffer
        self._initialized_buffer = True

    def __iter__(self):
        if not self._initialized_buffer:
            raise RuntimeError('Buffer or BufferedDataset have not been initialized')
        ixs, n = [], 0
        if self._iterdataset is None:
            self._iterdataset = iter(self.dataset)
        worker_info = get_worker_info()
        worker_id = None if worker_info is None else worker_info.id
        while n < self.dataset.worker_len():
            try:
                x = next(self._iterdataset)
            except StopIteration:
                if self._persistent_buffer:
                    # Continue loop
                    self._iterdataset = iter(self.dataset)
                    x = next(self._iterdataset)
                else:
                    # Stop loading data and start popping data from the buffer
                    break
            if len(self._buffer) == self._buffer_size:
                n += 1
                yield self._buffer.random_get_and_replace(x)
            else:
                self._buffer.append(x)

        self._buffer.send_done_looping()
        if not self._persistent_buffer:
            while not self._buffer.all_workers_done:
                sleep(.1)
            while len(self._buffer) > 0:
                yield self._buffer.pop()

    def __len__(self):
        return len(self.dataset)


class BufferedDataLoader(DataLoader):
    '''
    This dataloader wraps and sub-classes the usual PyTorch DataLoader and is
    intended for use with an iterable dataset where each worker has access to a
    different chunk of the data. Each worker loads a new datapoint form the
    disc and exchanges it with a random point from the buffer of the
    BufferedDataLoader.

    Args
    ----
        buffer_size (int)   The total buffer size (1 buffer for all workers)

        dataset (Dataset)   Typically an iterable dataset where each worker
            accesses a different chunk of the dataset. This dataset must
            implement the __len__ method.

        batch_size (int)    The output batch_size. This is also the batch_size
            used by the workers to import/read the data (see section 'Workflow
            of self.__iter__ below)

        persistent_buffer (bool)
            Since the first datapoints loaded by the worker only serve to fill
            the buffer, the workers will have finished their loop over the
            dataset before the dataloader has actually returned `len(dataset)`
            datapoints (actually `len(dataloader)` batches), i.e., it has not
            finished its 'epoch'. The `persistent_buffer` key says how it
            should finish that epoch: either by stopping the workers and
            popping data from the buffer until its empty (False); or by letting
            the workers start a new loop over the dataset and proceeding with
            the sample-and-replace process in the buffer as before. That way,
            the buffer will not need to get filled from scratch again in the
            next epoch. If `persistent_workers` is True, then, in the next
            epoch, the workers will resume their loop over the dataset from
            wherever they finished in the previous epoch.

        drop_last (bool)    Whether or not to drop the last incomplete batch.
            Irrelevant when `persistent_buffer` is True.
            (Remark: since each worker loads data in batches but only gets
            access to a specific subset of the data, the last batch from each
            worker may contain less than `batch_size` points.  However, these
            points get all passed separately to the global buffer. The actual
            output batches are constructed from this buffer, so that it's
            really only the last output batch for which this drop_last option
            is relevant.)

        **loader_args   Named arguments. Can be any of the arguments of the
            usual torch.utils.data.DataLoader.

    Comparison with torch.utils.data.BufferedShuffleDataset:
    --------------------------------------------------------

    PyTorch 1.8.1 provides a BufferedShuffleDataset that can f.ex. be called as
    follows:
        ```
        unshuffled_dataset = LMDBIterDataset(...)
        shuffled_dataset = BufferedShuffleDataset(unshuffled_dataset, buffer_size)
        dataloader = torch.utils.data.DataLoader(shuffled_dataset, num_workers, ...)
        ```
    Here, `shuffled_dataset` gets cloned in every worker (not
    `unshuffled_dataset` as in our BufferedDataLoader). That means that the
    buffer is specific to each worker, and contains / shuffles only the data
    loaded by that specific worker. I.e., the output batches will only contain
    shuffled data from 1 worker each, whereas our BufferedDataLoader contains
    and shuffles data accross all the workers.
    '''

    def __init__(self, buffer_size, dataset, batch_size, persistent_buffer=True,
                 num_workers=0, **kwargs):
        if 'persistent_workers' not in kwargs:
            kwargs['persistent_workers'] = persistent_buffer
        if persistent_buffer and (not kwargs['persistent_workers']):
            warnings.warn(
                'persistent_buffer is True but not `persistent_workers`. This may '
                'lead to duplicates. I recommend to make `persistent_worker` True '
                'if persistent_buffer is True')

        if 'shuffle' in kwargs and kwargs['shuffle'] == False:
            raise ValueError(f'You are using a BufferedDataLoader but passed argument shuffle=False.')

        self.buffer_size = min(len(dataset), buffer_size)
        self.persistent_buffer = True
        self.buffer = SafeBuffer(self.buffer_size, num_workers)
        self.dataset = BufferedDataset(dataset)

        super(BufferedDataLoader, self).__init__(
            self.dataset, batch_size=batch_size, num_workers=num_workers, **kwargs)

        self.dataset.initialize_buffer(self)

    def __iter__(self):
        self.buffer.reset()
        for data in super(BufferedDataLoader, self).__iter__():
            yield data

def list_collate_fn(l):
    # this must be declared at the top level when used on MacOS or Windows with
    # several workers (because of the way the spawning of new workers work)
    # See https://pytorch.org/docs/stable/data.html#platform-specific-behaviors
    return l

class BufferedDataLoader_old(object):
    '''
    This dataloader wraps the usual torch.utils.data.DataLoader, which is
    accessed internally stored in and accessed via self.loader. 

    Args
    ----
        buffer_size (int)   The total buffer size (1 buffer for all workers)

        dataset (Dataset)   Typically an iterable dataset where each worker
            accesses a different chunk of the dataset. This dataset must
            implement the __len__ method.

        batch_size (int)    The output batch_size. This is also the batch_size
            used by the workers to import/read the data (see section 'Workflow
            of self.__iter__ below)

        persistent_buffer (bool)
            Since the first datapoint loaded by the worker only serve to fill
            the buffer, the workers will have finished their loop over the
            dataset before the dataloader has actually returned `len(dataset)`
            datapoints (actually `len(dataloader)` batches), i.e., it has not
            finished its 'epoch'. The `persistent_buffer` key says how it
            should finish that epoch: either by stopping the workers and
            popping data from the buffer until its empty (False); or by letting
            the workers start a new loop over the dataset and proceeding with
            the sample-and-replace process in the buffer as before. That way,
            the buffer will not need to get filled from scratch again in the
            next epoch. If `persistent_workers` is True, then, in the next
            epoch, the workers will resume their loop over the dataset from
            wherever they finished in the previous epoch.

        drop_last (bool)    Whether or not to drop the last incomplete batch.
            Irrelevant when `persistent_buffer` is True.
            (Remark: since each worker loads data in batches but only gets
            access to a specific subset of the data, the last batch from each
            worker may contain less than `batch_size` points.  However, these
            points get all passed separately to the global buffer. The actual
            output batches are constructed from this buffer, so that it's
            really only the last output batch for which this drop_last option
            is relevant.)

        **loader_args   Named arguments. Can be any of the arguments of the
            usual torch.utils.data.DataLoader.

    Workflow of self.__iter__:
    --------------------------

    `self.__iter__` asks the workers from self.loader to provide/load the next
    data **batch**. Then it loops over each element of that batch to either
    fill `self.buffer` if it is not full, or to sample from it and replace the
    sampled point with the new element.
    This means in particular that:
        - the workers are called inside of `self.loader` and load data
          **batches** of size `batch_size` (same as the final output
          `batch_size`)
        - sampling from self.buffer and collating the sampled data for the
          output batch  happens in BufferedDataLoader, i.e., in the main
          process (not a worker thread), which is a bit slower than if it
          happened in the worker threads directly (as would be the case with a
          usual PyTorch DataLoader)
    
    Comparison with torch.utils.data.BufferedShuffleDataset:
    --------------------------------------------------------

    PyTorch 1.8.1 provides a BufferedShuffleDataset that can f.ex. be called as
    follows:
        ```
        unshuffled_dataset = LMDBIterDataset(...)
        shuffled_dataset = BufferedShuffleDataset(unshuffled_dataset, buffer_size)
        dataloader = torch.utils.data.DataLoader(shuffled_dataset, num_workers, ...)
        ```
    Here, its `shuffled_dataset` that gets cloned in every worker (not
    `unshuffled_dataset` as in our BufferedDataLoader). So that means that the
    buffer is specific to each worker, and contains and can shuffle only the
    data loaded by this specific worker. I.e., the output batches will only
    contain shuffled data from 1 worker each, whereas our BufferedDataLoader
    contains and shuffles data accross all the workers.
    '''

    def __init__(
            self, buffer_size, dataset, batch_size,
            persistent_buffer=True, drop_last=False, **loader_kwargs):
        # loader_kwargs should not contain the dataset, batch_size, drop_last
        # and collate_fn argument
        loader_collate_fn = list_collate_fn
        if 'collate_fn' in loader_kwargs:
            self.collate_fn = loader_kwargs['collate_fn']
        else:
            self.collate_fn = default_collate
        loader_kwargs['collate_fn'] = loader_collate_fn

        if 'generator' in loader_kwargs:
            self.generator = loader_kwargs['generator']
        else:
            self.generator = None

        if 'persistent_workers' not in loader_kwargs:
            loader_kwargs['persistent_workers'] = persistent_buffer
        if persistent_buffer and (not loader_kwargs['persistent_workers']):
            warnings.warn(
                'persistent_buffer is True but not `persistent_workers`. This may '
                'lead to duplicates. I recommend to make `persistent_worker` True '
                'if persistent_buffer is True')

        if 'shuffle' in loader_kwargs and loader_kwargs['shuffle'] == False:
            raise ValueError(f'You are using a BufferedDataLoader but passed argument shuffle=False.')

        assert buffer_size >= batch_size, 'buffer_size must be >= batch_size'
        # NB: drop_last is always set to False in the internal data-loader. The
        # `drop_last`argument is only used for the BufferedDataLoader
        self.loader = DataLoader(
            dataset, batch_size, drop_last=False, **loader_kwargs)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.persistent_buffer = persistent_buffer
        self.buffer_size = min(buffer_size, len(self.loader.dataset))
        self.buffer = []
        self.batch = []

    def __iter__(self):
        ixs, n = [], 0
        iterloader = iter(self.loader)
        # for data_batch in self.loader:
        while n < len(self):
            try:
                data_batch = next(iterloader)
            except StopIteration:
                iterloader = iter(self.loader)
                if self.persistent_buffer:
                    # Continue loading the buffer and sampling from it
                    data_batch = next(iterloader)
                else:
                    # Break and start flushing the buffer
                    break
            ixs = torch.randint(
                high=self.buffer_size, size=(len(data_batch),),
                dtype=torch.int64, generator=self.generator).tolist()
            for data in data_batch:
                if len(self.buffer) == self.buffer_size:
                    ix = ixs.pop()
                    self.batch.append(self.buffer[ix])
                    self.buffer[ix] = data
                    if len(self.batch) == self.batch_size:
                        n += 1
                        yield self.collate_fn(self.batch)
                        self.batch = []
                else:
                    self.buffer.append(data)

        # Flush the buffer if not persistent_buffer
        if not self.persistent_buffer:
            random.shuffle(self.buffer)
            while self.buffer:
                self.batch.append(self.buffer.pop())
                if len(self.batch) == self.batch_size:
                    yield self.collate_fn(self.batch)
                    self.batch = []
            if len(self.batch) > 0 and not self.drop_last:
                yield self.collate_fn(self.batch)
            self.batch = []

    def __len__(self):
        return len(self.loader)
