# -*- coding: utf-8 -*-
# File: base.py
# Code copied from tensorpack.dataflow, Revision e7c32ae4

import threading
from abc import ABCMeta, abstractmethod
import six

# from ..utils.utils import get_rng

# __all__ = ['DataFlow', 'ProxyDataFlow', 'RNGDataFlow', 'DataFlowTerminated']

import numpy as np
import os
import logging
import lmdb
from contextlib import contextmanager
from datetime import datetime

# from ..utils import logger
# from ..utils.serialize import loads
# from ..utils.timer import timed_operation
from pickle import loads
# from ..utils.utils import get_tqdm
# from .base import DataFlowReentrantGuard, RNGDataFlow

# __all__ = ['HDF5Data', 'LMDBData', 'LMDBDataDecoder', 'CaffeLMDB', 'SVMLightData']

logger = logging.Logger('dataflow')


## Copied from dataflow.utils

_RNG_SEED = None

def get_rng(obj=None):
    """
    Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)


def get_tqdm_kwargs(**kwargs):
    """
    Return default arguments to be used with tqdm.

    Args:
        kwargs: extra arguments to be used.
    Returns:
        dict:
    """
    default = dict(
        smoothing=0.5,
        dynamic_ncols=True,
        ascii=True,
        bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_noinv_fmt}]'
    )

    try:
        # Use this env var to override the refresh interval setting
        interval = float(os.environ['TENSORPACK_PROGRESS_REFRESH'])
    except KeyError:
        interval = _pick_tqdm_interval(kwargs.get('file', sys.stderr))

    default['mininterval'] = interval
    default.update(kwargs)
    return default


def get_tqdm(*args, **kwargs):
    """ Similar to :func:`tqdm.tqdm()`,
    but use tensorpack's default options to have consistent style. """
    return tqdm(*args, **get_tqdm_kwargs(**kwargs))

@contextmanager
def timed_operation(msg, log_start=False):
    """
    Surround a context with a timer.

    Args:
        msg(str): the log to print.
        log_start(bool): whether to print also at the beginning.

    Example:
        .. code-block:: python

            with timed_operation('Good Stuff'):
                time.sleep(1)

        Will print:

        .. code-block:: python

            Good stuff finished, time:1sec.
    """
    assert len(msg)
    if log_start:
        logger.info('Start {} ...'.format(msg))
    start = timer()
    yield
    msg = msg[0].upper() + msg[1:]
    logger.info('{} finished, time:{:.4f} sec.'.format(
        msg, timer() - start))


# copied from dataflow.base
class DataFlowTerminated(BaseException):
    """
    An exception indicating that the DataFlow is unable to produce any more
    data, i.e. something wrong happened so that calling :meth:`get_data`
    cannot give a valid iterator any more.
    In most DataFlow this will never be raised.
    """
    pass



class DataFlowReentrantGuard(object):
    """
    A tool to enforce non-reentrancy.
    Mostly used on DataFlow whose :meth:`get_data` is stateful,
    so that multiple instances of the iterator cannot co-exist.
    """
    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._succ = self._lock.acquire(False)
        if not self._succ:
            raise threading.ThreadError("This DataFlow is not reentrant!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        return False


class DataFlowMeta(ABCMeta):
    """
    DataFlow uses "__iter__()" and "__len__()" instead of
    "get_data()" and "size()". This add back-compatibility.
    """
    def __new__(mcls, name, bases, namespace, **kwargs):

        def hot_patch(required, existing):
            if required not in namespace and existing in namespace:
                namespace[required] = namespace[existing]

        hot_patch('__iter__', 'get_data')
        hot_patch('__len__', 'size')

        return ABCMeta.__new__(mcls, name, bases, namespace, **kwargs)


@six.add_metaclass(DataFlowMeta)
class DataFlow(object):
    """ Base class for all DataFlow """

    @abstractmethod
    def __iter__(self):
        """
        * A dataflow is an iterable. The :meth:`__iter__` method should yield a list or dict each time.
          Note that dict is **partially** supported at the moment: certain dataflow does not support dict.

        * The :meth:`__iter__` method can be either finite (will stop iteration) or infinite
          (will not stop iteration). For a finite dataflow, :meth:`__iter__` can be called
          again immediately after the previous call returned.

        * For many dataflow, the :meth:`__iter__` method is non-reentrant, which means for an dataflow
          instance ``df``, :meth:`df.__iter__` cannot be called before the previous
          :meth:`df.__iter__` call has finished (iteration has stopped).
          When a dataflow is non-reentrant, :meth:`df.__iter__` should throw an exception if
          called before the previous call has finished.
          For such non-reentrant dataflows, if you need to use the same dataflow in two places,
          you need to create two dataflow instances.

        Yields:
            list/dict: The datapoint, i.e. list/dict of components.
        """


    def __len__(self):
        """
        * A dataflow can optionally implement :meth:`__len__`. If not implemented, it will
          throw :class:`NotImplementedError`.

        * It returns an integer representing the size of the dataflow.
          The return value **may not be accurate or meaningful** at all.
          When saying the length is "accurate", it means that
          :meth:`__iter__` will always yield this many of datapoints before it stops iteration.

        * There could be many reasons why :meth:`__len__` is inaccurate.
          For example, some dataflow has dynamic size, if it throws away datapoints on the fly.
          Some dataflow mixes the datapoints between consecutive passes over
          the dataset, due to parallelism and buffering.
          In this case it does not make sense to stop the iteration anywhere.

        * Due to the above reasons, the length is only a rough guidance.
          And it's up to the user how to interpret it.
          Inside tensorpack it's only used in these places:

          + A default ``steps_per_epoch`` in training, but you probably want to customize
            it yourself, especially when using data-parallel trainer.
          + The length of progress bar when processing a dataflow.
          + Used by :class:`InferenceRunner` to get the number of iterations in inference.
            In this case users are **responsible** for making sure that :meth:`__len__` is "accurate".
            This is to guarantee that inference is run on a fixed set of images.

        Returns:
            int: rough size of this dataflow.

        Raises:
            :class:`NotImplementedError` if this DataFlow doesn't have a size.
        """
        raise NotImplementedError()


    def reset_state(self):
        """
        * The caller must guarantee that :meth:`reset_state` should be called **once and only once**
          by the **process that uses the dataflow** before :meth:`__iter__` is called.
          The caller thread of this method should stay alive to keep this dataflow alive.

        * It is meant for certain initialization that involves processes,
          e.g., initialize random number generators (RNG), create worker processes.

          Because it's very common to use RNG in data processing,
          developers of dataflow can also subclass :class:`RNGDataFlow` to have easier access to
          a properly-initialized RNG.

        * A dataflow is not fork-safe after :meth:`reset_state` is called (because this will violate the guarantee).
          There are a few other dataflows that are not fork-safe anytime, which will be mentioned in the docs.

        * You should take the responsibility and follow the above guarantee if you're the caller of a dataflow yourself
          (either when you're using dataflow outside of tensorpack, or if you're writing a wrapper dataflow).

        * Tensorpack's built-in forking dataflows (:class:`MultiProcessRunner`, :class:`MultiProcessMapData`, etc)
          and other component that uses dataflows (:class:`InputSource`)
          already take care of the responsibility of calling this method.
        """
        pass


    # These are the old (overly verbose) names for the methods:
    def get_data(self):
        return self.__iter__()

    def size(self):
        return self.__len__()



class RNGDataFlow(DataFlow):
    """ A DataFlow with RNG"""

    rng = None
    """
    ``self.rng`` is a ``np.random.RandomState`` instance that is initialized
    correctly (with different seeds in each process) in ``RNGDataFlow.reset_state()``.
    """

    def reset_state(self):
        """ Reset the RNG """
        self.rng = get_rng(self)


class LMDBData(RNGDataFlow):
    """
    Read a LMDB database and produce (k,v) raw bytes pairs.
    The raw bytes are usually not what you're interested in.
    You might want to use
    :class:`LMDBDataDecoder` or apply a
    mapper function after :class:`LMDBData`.
    """
    def __init__(self, lmdb_path, shuffle=True, keys=None):
        """
        Args:
            lmdb_path (str): a directory or a file.
            shuffle (bool): shuffle the keys or not.
            keys (list[str] or str): list of str as the keys, used only when shuffle is True.
                It can also be a format string e.g. ``{:0>8d}`` which will be
                formatted with the indices from 0 to *total_size - 1*.

                If not given, it will then look in the database for ``__keys__`` which
                :func:`LMDBSerializer.save` used to store the list of keys.
                If still not found, it will iterate over the database to find
                all the keys.
        """
        self._lmdb_path = lmdb_path
        self._shuffle = shuffle

        self._open_lmdb()
        self._size = self._txn.stat()['entries']
        self._set_keys(keys)
        logger.info("Found {} entries in {}".format(self._size, self._lmdb_path))

        # Clean them up after finding the list of keys, since we don't want to fork them
        self._close_lmdb()


    def _set_keys(self, keys=None):
        def find_keys(txn, size):
            logger.warn("Traversing the database to find keys is slow. Your should specify the keys.")
            keys = []
            with timed_operation("Loading LMDB keys ...", log_start=True), \
                    get_tqdm(total=size) as pbar:
                for k in self._txn.cursor():
                    assert k[0] != b'__keys__'
                    keys.append(k[0])
                    pbar.update()
            return keys

        self.keys = self._txn.get(b'__keys__')
        if self.keys is not None:
            self.keys = loads(self.keys)
            self._size -= 1     # delete this item

        if self._shuffle:   # keys are necessary when shuffle is True
            if keys is None:
                if self.keys is None:
                    self.keys = find_keys(self._txn, self._size)
            else:
                # check if key-format like '{:0>8d}' was given
                if isinstance(keys, six.string_types):
                    self.keys = map(lambda x: keys.format(x), list(np.arange(self._size)))
                else:
                    self.keys = keys

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

    def reset_state(self):
        self._guard = DataFlowReentrantGuard()
        super(LMDBData, self).reset_state()
        self._open_lmdb()  # open the LMDB in the worker process

    def __len__(self):
        return self._size

    def __iter__(self):
        with self._guard:
            if not self._shuffle:
                c = self._txn.cursor()
                for k, v in c:
                    if k != b'__keys__':
                        yield [k, v]
            else:
                self.rng.shuffle(self.keys)
                for k in self.keys:
                    v = self._txn.get(k)
                    yield [k, v]
