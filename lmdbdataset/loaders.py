import os
import logging
import lmdb
from pickle import loads
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from random import shuffle
from torch.utils.data import get_worker_info, Dataset, IterableDataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch.distributed as dist

# from .dataflow import LMDBData

# try:
#     from dataflow import LMDBData  # only tensorpack.dataflow needed
# except ImportError:
#     from tensorpack.dataflow import LMDBData

_logger = logging.getLogger('loader')

class LMDBPrefetcher(object):
    def __init__(self, lmdb_path, keys=None):
        """
        This class is adapted from the LMDBData class in tensorpack.dataflow.
        It contains the actual LMDB-file reader and will be used by LMDBDataset
        (which directly invokes __getitem__) and LMDBIterableDataset. It loads
        the pickled / serialized images from the lmdb file und unpickles /
        deserializes them.

        Args:
            lmdb_path (str): a directory or a file.
            keys (list[str] or str): list of str as the keys.
                It can also be a format string e.g. ``{:0>8d}`` which will be
                formatted with the indices from 0 to *total_size - 1*. It will
                be ignored if the lmdb file contains a `__keys__` argument.
                If None, then the lmdb must contain a `__keys__` argument. We
                not automatically reconstructing these keys.
        """
        self._lmdb_path = lmdb_path
        self._open_lmdb()
        self._size = self._txn.stat()['entries']
        self._set_keys(keys)
        _logger.info("Found {} entries in {}".format(self._size, self._lmdb_path))
        # Clean them up after finding the list of keys, since we don't want to fork them
        self._close_lmdb()

    def _set_keys(self, keys=None):
        self.keys = self._txn.get(b'__keys__')
        if self.keys is not None:
            self.keys = loads(self.keys)
            self._size -= 1     # delete this item
            if keys is not None:
                logging.warn(
                    'Found keys in lmdb file. Ignoring keys passed as argument.')
        elif keys is not None:
            if isinstance(keys, six.string_types):
                self.keys = map(lambda x: keys.format(x), list(np.arange(self._size)))
            else:
                self.keys = keys
        else:
            raise ValueError(
                f'No `__keys__` entry found in lmdb dataset and no keys passed as argument.')
        assert self._size == len(self.keys), (
            'len(self.keys) does not match length found using '
            'lmdb.transaction.stat()')

    def _open_lmdb(self):
        self._lmdb = lmdb.open(self._lmdb_path,
                               subdir=os.path.isdir(self._lmdb_path),
                               readonly=True, lock=False, readahead=True,
                               map_size=1099511627776 * 2, max_readers=100)
        self._txn = self._lmdb.begin()

    def _close_lmdb(self):
        self._lmdb.close()
        del self._lmdb
        del self._txn

    def reset(self):
        self._open_lmdb()

    def __len__(self):
        return self._size

    def __getitem__(self, ix):
        self._open_lmdb()
        k = self.keys[ix]
        v = self._txn.get(k)
        return loads(v)


class _LMDBDataset(object):
    '''
        A map-style dataset that opens an LMDB file and randomly loads its
        files/images using the LMDB's dictionnary/key-value based access.
        Note that the LMDB file is opened only once (actually twice: at
        initialization, and then after worker-creation).
        
        Rmk: This class will be reused by LMDBIterDataset. Therefore, we do not
        want to inherit from PyTorch's Dataset class here, because of possible
        conflicts with PyTorch's IterableDataset class later.

    Arguments
    ---------
        root (str)  Path to LMDB dataset folder (should contain train.lmdb and
            val.lmdb
        split (str) Must be 'train' or 'val'
        img_type (str)  'numpy' or 'pil'. Should be the same than argument
            used in create_lmdb(.) function
        return_type (str)   'pil', 'numpy' or 'torch'. If 'pil',
            then img_type must also be 'pil'
    '''

    def __init__(self, root, split='train', transform=None,
                 transform_target=None, img_type='numpy', return_type='torch',):

        fname = 'train.lmdb' if split == 'train' else 'val.lmdb'
        lmdb_path = os.path.expanduser(os.path.join(root, fname))
        self.lmdb_path = lmdb_path
        assert os.path.exists(self.lmdb_path), f'Non existing path {self.lmdb_path}'

        assert img_type in ['numpy', 'pil'], f'Unknonwn img_type {img_type}'
        if return_type == 'pil' and img_type == 'numpy':
            raise ValueError(
                f"return_type {return_type} incompatible with img_type 'numpy'. "
                f"Use return_type 'torch'")
        self.img_type = img_type
        self.return_type = return_type
        self.prefetcher = LMDBPrefetcher(self.lmdb_path)
        self.transform = transform
        self.transform_target = transform_target
        self._initialized_worker = False

    def __getitem__(self, ix):
        if not self._initialized_worker:
            self.initialize_worker()
        img, target = self.prefetcher[ix]
        if self.img_type == 'pil':
            img = Image.open(BytesIO(img)).convert('RGB')
            if self.return_type in {'numpy','torch'}:
                # img = np.asarray(img) would be better, but bc it doesn't
                # copy the tensor data. But Image returns an img that is
                # read-only, so that torch trows an anooying warning.
                # np.array doesn't seem slower.
                img = np.array(img)
                img = np.rollaxis(img, 2)  # HWC to CHW
        else:
            ValueError('img_type must be jpeg or numpy')
        if self.return_type == 'torch':
            img = torch.tensor(img).contiguous()
            target = torch.tensor(target)
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.tensor(-1, dtype=torch.long)
        if self.transform_target is not None:
            target = self.transform_target(target)
        return img, target

    def initialize_worker(self):
        self.prefetcher.reset()
        self._initialized_worker = True

    def __len__(self):
        return len(self.prefetcher)

class LMDBDataset(_LMDBDataset, Dataset):
    '''
        This is the same as _LMBDDataset, but with the inheritance from
        PyTorch's Dataset class added. We add it here and not in _LMDBDataset
        so that the iterable LMDBIterDataset can inherit from PyTorch's
        IterableDataset class only, and avoid conflicts with the Dataset class.
    '''
    pass

class LMDBIterDataset(_LMDBDataset, IterableDataset):
    '''
        An iterable dataset that opens an LMDB file and reads it in the order
        of its keys (which should be the order of storage). If multiple workers
        are used, each worker gets a different chunk of the dataset.

    Arguments
    ---------
        root (str)  Path to LMDB dataset folder (should contain train.lmdb and
            val.lmdb
        split (str) Must be 'train' or 'val'
        img_type (str)  'numpy' or 'pil'. Should be the same than argument
            used in create_lmdb(.) function
        return_type (str)   'pil', 'numpy' or 'torch'. If 'pil',
            then img_type must also be 'pil'
        distributed (bool)  Whether this dataset is used in a distributed
            context (e.g., with DistributedDataParallel). Defaults to False,
            except if rank is not None or world_size > 1.
        rank_size (int)  Rank of the process, when using this dataset accross
            multiple processes (e.g., in conjunction with the
            DistributedDataParallel module). If None, the rank will be
            retrieved from the rand in the current group.
        world_size (int) Number of processes, when using multiple processes. If
            None, world_size will be retrieved from the current group.

    NB: LMDBDataset has no handling for distributed processes, because for
    map-style datasets, the distribution of idx accross processes should be
    handled by passing PyTorch's DistributedSampler to the DataLoader.
    '''

    def __init__(self, root, split='train', transform=None,
                 transform_target=None, img_type='numpy', return_type='torch',
                 distributed=None, rank=None, world_size=None):

        super(LMDBIterDataset, self).__init__(
            root, split, transform, transform_target, img_type, return_type)

        self._distributed = distributed
        if self._distributed is None:
            self._distributed = (rank is not None) or (
                (world_size is not None) and world_size > 1)

        self.start = self.worker_start = self.rank_start = 0
        self.end = self.worker_end = self.rank_end = len(self.prefetcher)
        # Contrary to workers which get initialized by the dataloader, the rank
        # is already defined at initialization of the dataset. Hence, we define
        # the rank here (which is usefull for the dataloader to get the correct
        # dataset length) and initialize the worker specific parameters in the
        # first iteration __iter__
        self.initialize_rank(rank, world_size)
        self._initialized_worker = False

    def initialize_rank(self, rank, world_size):
        # Determine rank_start / rank_end
        if self._distributed:
            self._rank = dist.get_rank() if rank is None else rank
            self._world_size = dist.get_world_size() if world_size is None else world_size
            per_rank = int(np.ceil((self.end - self.start) / float(self._world_size)))
            self.rank_start = self.start + self._rank * per_rank
            self.rank_end = min(self.rank_start + per_rank, self.end)
        else:
            self._rank = 0
            self._world_size = 1

    def initialize_worker(self):
        '''
            When loading this iterative dataset with multiple workers, then
            each worker gets a different chunk of the dataset to load, and
            loads it sequentially.
        '''
        self.prefetcher.reset()

        # Determine worker_start / worker_end in the current rank/process
        worker_info = get_worker_info()
        if worker_info is not None:  # multi-process loading: we are in a worker
            per_worker = int(np.ceil((self.rank_end - self.rank_start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            self.worker_start = self.rank_start + worker_id * per_worker
            self.worker_end = min(self.worker_start + per_worker, self.rank_end)
            _logger.debug(
                f"Initialized worker {worker_id} in rank {self._rank}.\t"
                f"Start: {self.worker_start}\t"
                f"End: {self.worker_end}")

        self._initialized_worker = True

    def __iter__(self):
        if not self._initialized_worker:
            self.initialize_worker()
        for ix in range(self.worker_start, self.worker_end):
            yield super(LMDBIterDataset,self).__getitem__(ix)

    def __len__(self):
        # returns the length of the chunk seen by this process
        return self.rank_end - self.rank_start

    def full_len(self):
        # returns the full length of the dataset
        return self.end - self.start

    def worker_len(self):
        # returns the length of the chunk seen by this worker
        return self.worker_end - self.worker_start


def list_collate_fn(l):
    # this must be declared at the top level when used on MacOS or Windows with
    # several workers (because of the way the spawning of new workers work)
    # See https://pytorch.org/docs/stable/data.html#platform-specific-behaviors
    return l

class BufferedDataLoader(DataLoader):
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
            self, dataset, buffer_size, batch_size,
            persistent_buffer=True, drop_last=False, **loader_kwargs):
        # loader_kwargs should not contain the dataset, batch_size, drop_last
        # and collate_fn argument
        loader_collate_fn = list_collate_fn
        if 'collate_fn' in loader_kwargs:
            self.buffer_collate_fn = loader_kwargs['collate_fn']
        else:
            self.buffer_collate_fn = default_collate
        loader_kwargs['collate_fn'] = loader_collate_fn

        if 'generator' in loader_kwargs:
            self.generator = loader_kwargs['generator']
        else:
            self.generator = None

        if 'persistent_workers' not in loader_kwargs:
            loader_kwargs['persistent_workers'] = persistent_buffer
        if persistent_buffer and (not loader_kwargs['persistent_workers']):
            _logger.warning(
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

        super(BufferedDataLoader, self).__init__(
            dataset, batch_size, drop_last=False, **loader_kwargs)

        # making the buffer_batch_size = batch_size of workers. If we want to
        # change this one day, then we need to recode the __len__ method with
        # buffer_batch_size instead of batch_size
        self.buffer_batch_size = self.batch_size
        self.buffer_drop_last = drop_last
        self.persistent_buffer = persistent_buffer
        self.buffer_size = min(buffer_size, len(self.dataset))
        self.buffer = []
        self.batch = []

    def __iter__(self):
        ixs, n = [], 0
        iterloader = super(BufferedDataLoader,self).__iter__()
        # for data_batch in self.loader:
        while n < len(self):
            try:
                data_batch = next(iterloader)
            except StopIteration:
                if self.persistent_buffer:
                    # Continue loading the buffer and sampling from it
                    iterloader = super(BufferedDataLoader,self).__iter__()
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
                    if len(self.batch) == self.buffer_batch_size:
                        n += 1
                        yield self.buffer_collate_fn(self.batch)
                        self.batch = []
                else:
                    self.buffer.append(data)

        # Flush the buffer if not persistent_buffer
        if not self.persistent_buffer:
            shuffle(self.buffer)
            while self.buffer:
                self.batch.append(self.buffer.pop())
                if len(self.batch) == self.buffer_batch_size:
                    yield self.buffer_collate_fn(self.batch)
                    self.batch = []
            if len(self.batch) > 0 and not self.buffer_drop_last:
                yield self.buffer_collate_fn(self.batch)
            self.batch = []



# ##############################################################
# ### ALTERNATIVE BUFFERED DATALOADER USING A SHARED BUFFER ####
# ##############################################################
# 
# '''
#     In the rest of this file, we implement an alterantive BufferedDataLoader,
#     which uses a shared buffer. Each worker loads data, samples a random idx of
#     that buffer and swaps its data with the data contained at that idx. That
#     way, the worker can also do the batching (collation of data-points into
#     batches), which, in the implementatino above is handled by the main
#     program. However, this requires additionnal communication time, which, in
#     the end, makes this slower than the implementation above. Also, it doesn't
#     seem to work when using PyTorch's DistributedParallel.
# '''
# 
# from time import sleep
# import torch.multiprocessing as mp
# from torch.multiprocessing import Lock
# 
# 
# class SafeBuffer(object):
#     '''
#         This class provides a thread-safe buffer to be used by the
#         `BufferedDataLoader`, which will pass this buffer to every of its
#         workers (the dataset-loading sub-processes).
#     '''
# 
#     def __init__(self, buffer_size: int, num_workers: int):
#         manager = mp.Manager()
#         self._list = manager.list()
#         self._lock = manager.Lock()
#         self._buffer_size = buffer_size
#         self._num_workers = num_workers
#         self._num_workers_done_looping = mp.Value('i', 0)
#         self._ixs = []
#         _logger.debug('Created safe buffer. There should be 1 buffer per dataloader.')
# 
#     def append(self, x):
#         self._list.append(x)
# 
#     def pop(self):
#         return self._list.pop()
# 
#     def random_get_and_replace(self, x):
#         self._lock.acquire()
#         ix = self.sample_random_ix()
#         out = self._list[ix]
#         self._list[ix] = x
#         self._lock.release()
#         return out
# 
#     def sample_random_ix(self):
#         if len(self._ixs) == 0:
#             self._ixs = torch.randint(high=len(self), size=(100,),
#                                       dtype=torch.int64).tolist()
#         return self._ixs.pop()
# 
#     @property
#     def all_workers_done(self):
#         return (self._num_workers_done_looping.value >= self._num_workers)
# 
#     def send_done_looping(self):
#         with self._num_workers_done_looping.get_lock():
#             self._num_workers_done_looping.value += 1
#             if self._num_workers_done_looping == self._num_workers:
#                 shuffle(self._list)
# 
#     def reset(self):
#         self._num_workers_done_looping.value = 0
# 
#     def __len__(self):
#         return len(self._list)
# 
# 
# class BufferedDataset(IterableDataset):
#     '''
#         This class is a wrapper for iterable datasets that is used by the
#         BufferedDataLoader to handle access to a joint buffer between
#         different workers.
#     '''
#     def __init__(self, dataset):
#         super(BufferedDataset, self).__init__()
#         self.dataset = dataset
#         self._initialized_buffer = False
#         self._iterdataset = None
# 
#     def initialize_buffer(self, buffered_dataloader): # buf, buffer_size, persistent_buffer):
#         # The buffer comes from the loader and is the same for all workers
#         self._buffer = buffered_dataloader.buffer
#         self._buffer_size = buffered_dataloader.buffer_size
#         self._persistent_buffer = buffered_dataloader.persistent_buffer
#         self._initialized_buffer = True
# 
#     def __iter__(self):
#         if not self._initialized_buffer:
#             raise RuntimeError('Buffer or BufferedDataset have not been initialized')
#         ixs, n = [], 0
#         if self._iterdataset is None:
#             self._iterdataset = iter(self.dataset)
#         worker_info = get_worker_info()
#         worker_id = None if worker_info is None else worker_info.id
#         while n < self.dataset.worker_len():
#             try:
#                 x = next(self._iterdataset)
#             except StopIteration:
#                 if self._persistent_buffer:
#                     # Continue loop
#                     self._iterdataset = iter(self.dataset)
#                     x = next(self._iterdataset)
#                 else:
#                     # Stop loading data and start popping data from the buffer
#                     break
#             if len(self._buffer) == self._buffer_size:
#                 n += 1
#                 yield self._buffer.random_get_and_replace(x)
#             else:
#                 self._buffer.append(x)
# 
#         self._buffer.send_done_looping()
#         if not self._persistent_buffer:
#             while not self._buffer.all_workers_done:
#                 sleep(.1)
#             while len(self._buffer) > 0:
#                 yield self._buffer.pop()
# 
#     def __len__(self):
#         return len(self.dataset)
# 
# 
# class BufferedDataLoader2(DataLoader):
#     '''
#     This dataloader wraps and sub-classes the usual PyTorch DataLoader and is
#     intended for use with an iterable dataset where each worker has access to a
#     different chunk of the data. Each worker loads a new datapoint form the
#     disc and exchanges it with a random point from the buffer of the
#     BufferedDataLoader.
# 
#     Args
#     ----
#         buffer_size (int)   The total buffer size (1 buffer for all workers)
# 
#         dataset (Dataset)   Typically an iterable dataset where each worker
#             accesses a different chunk of the dataset. This dataset must
#             implement the __len__ method.
# 
#         batch_size (int)    The output batch_size. This is also the batch_size
#             used by the workers to import/read the data (see section 'Workflow
#             of self.__iter__ below)
# 
#         persistent_buffer (bool)
#             Since the first datapoints loaded by the worker only serve to fill
#             the buffer, the workers will have finished their loop over the
#             dataset before the dataloader has actually returned `len(dataset)`
#             datapoints (actually `len(dataloader)` batches), i.e., it has not
#             finished its 'epoch'. The `persistent_buffer` key says how it
#             should finish that epoch: either by stopping the workers and
#             popping data from the buffer until its empty (False); or by letting
#             the workers start a new loop over the dataset and proceeding with
#             the sample-and-replace process in the buffer as before. That way,
#             the buffer will not need to get filled from scratch again in the
#             next epoch. If `persistent_workers` is True, then, in the next
#             epoch, the workers will resume their loop over the dataset from
#             wherever they finished in the previous epoch.
# 
#         drop_last (bool)    Whether or not to drop the last incomplete batch.
#             Irrelevant when `persistent_buffer` is True.
#             (Remark: since each worker loads data in batches but only gets
#             access to a specific subset of the data, the last batch from each
#             worker may contain less than `batch_size` points.  However, these
#             points get all passed separately to the global buffer. The actual
#             output batches are constructed from this buffer, so that it's
#             really only the last output batch for which this drop_last option
#             is relevant.)
# 
#         **loader_args   Named arguments. Can be any of the arguments of the
#             usual torch.utils.data.DataLoader.
# 
#     Comparison with torch.utils.data.BufferedShuffleDataset:
#     --------------------------------------------------------
# 
#     PyTorch 1.8.1 provides a BufferedShuffleDataset that can f.ex. be called as
#     follows:
#         ```
#         unshuffled_dataset = LMDBIterDataset(...)
#         shuffled_dataset = BufferedShuffleDataset(unshuffled_dataset, buffer_size)
#         dataloader = torch.utils.data.DataLoader(shuffled_dataset, num_workers, ...)
#         ```
#     Here, `shuffled_dataset` gets cloned in every worker (not
#     `unshuffled_dataset` as in our BufferedDataLoader). That means that the
#     buffer is specific to each worker, and contains / shuffles only the data
#     loaded by that specific worker. I.e., the output batches will only contain
#     shuffled data from 1 worker each, whereas our BufferedDataLoader contains
#     and shuffles data accross all the workers.
#     '''
# 
#     def __init__(self, dataset, buffer_size, batch_size, persistent_buffer=True,
#                  num_workers=0, **kwargs):
#         if 'persistent_workers' not in kwargs:
#             kwargs['persistent_workers'] = persistent_buffer
#         if persistent_buffer and (not kwargs['persistent_workers']):
#             _logger.warning(
#                 'persistent_buffer is True but not `persistent_workers`. This may '
#                 'lead to duplicates. I recommend to make `persistent_worker` True '
#                 'if persistent_buffer is True')
# 
#         if 'shuffle' in kwargs and kwargs['shuffle'] == False:
#             raise ValueError(f'You are using a BufferedDataLoader but passed argument shuffle=False.')
# 
#         self.buffer_size = min(len(dataset), buffer_size)
#         self.persistent_buffer = True
#         self.buffer = SafeBuffer(self.buffer_size, num_workers)
#         self.dataset = BufferedDataset(dataset)
# 
#         super(BufferedDataLoader, self).__init__(
#             self.dataset, batch_size=batch_size, num_workers=num_workers, **kwargs)
# 
#         self.dataset.initialize_buffer(self)
# 
#     def __iter__(self):
#         self.buffer.reset()
#         for data in super(BufferedDataLoader, self).__iter__():
#             yield data
