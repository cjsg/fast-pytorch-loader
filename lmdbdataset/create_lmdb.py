import os
import platform
import lmdb
import logging
import argparse
import numpy as np

from PIL import Image
from pickle import dumps
from tqdm import tqdm

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def open_to_raw(path_to_file):
    with open(path_to_file, 'rb') as f:
        raw = f.read()  # return bytes
    raw_bytearray = bytearray(raw)  # returns bytearray (=mutable bytes)
    return np.asarray(raw_bytearray, dtype='uint8')  # converts to np.ndarray of bytes

def open_to_numpy(path_to_file):
    img = Image.open(path_to_file).convert('RGB')
    np_img = np.array(img, dtype='uint8')
    np_img = np_img.transpose((2,0,1))  # HWC to CHW (default PyTorch format)
    return np_img

def put_or_grow(txn, key, value):
    # Copied from LMDBSerializer.save in tensorpack.dataflow
    # Put data into lmdb, and doubling the size if full.
    # Ref: https://github.com/NVIDIA/DIGITS/pull/209/files
    try:
        txn.put(key, value)
        return txn
    except lmdb.MapFullError:
        pass
    txn.abort()
    curr_size = db.info()['map_size']
    new_size = curr_size * 2
    logger.info("Doubling LMDB map_size to {:.2f}GB".format(new_size / 10**9))
    db.set_mapsize(new_size)
    txn = db.begin(write=True)
    txn = put_or_grow(txn, key, value)
    return txn

class ImageFolderToBinary(ImageFolder):
    def __init__(self, root, save_as='raw'):
        '''
        save_as (str) : 'raw' or 'numpy'
        '''
        root = os.path.expanduser(root)
        if save_as == 'raw':
            loader = open_to_raw
        elif save_as == 'numpy':
            loader = open_to_numpy
        else:
            raise ValueError(f'Unknown save_as format {save_as}.')

        logger.info('Searching for images...')
        super(ImageFolderToBinary, self).__init__(root, loader=loader)
        logger.info(
            f'Found {len(self)} images split accross {len(self.classes)} '
            f'categories in {root}.')


def create_lmdb(from_dir, to_dir, name, save_as='raw', shuffle=True, workers=0,
                write_freq=5000):
    '''
    from_dir (str)  root of dataset containing the image-folder (organized as
                    for PyTorch's standard torchvision.datasets.ImageFolder)
    to_dir (str)    target directory where to save the lmdb file
    name (str)      name of the lmdb file (without extension). File be:
                    `to_dir/name.lmdb`
    shuffle (bool)  whether to shuffle the dataset/image-folder. We advise True
                    for the training set, and False for the validation/test
                    sets. Default: True.
    workers (int)   number of workers to use
    write_freq (int)    how many images to process before writing to harddrive.
    '''

    from_dir = os.path.expanduser(from_dir)
    to_dir = os.path.expanduser(to_dir)
    to_file = os.path.join(to_dir, name+'.lmdb')

    assert os.path.isdir(from_dir)
    assert os.path.isdir(to_dir)
    assert not os.path.isfile(to_file), "LMDB file {} exists!".format(to_file)
    assert save_as in ['raw', 'numpy']

    dataset = ImageFolderToBinary(from_dir, save_as)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=shuffle, num_workers=workers)
        
    # The code below is copied from tensorpack.dataflow, LMDBSerializer.save
    # It's OK to use super large map_size on Linux, but not on other platforms
    # See: https://github.com/NVIDIA/DIGITS/issues/206
    map_size = 1099511627776 * 2 if platform.system() == 'Linux' else 128 * 10**6
    db = lmdb.open(to_file, subdir=False,
                   map_size=map_size, readonly=False,
                   meminit=False, map_async=True)    # need sync() at the end

    logger.info(f'Starting to serialize datataset from {from_dir} to {to_file}')
    with tqdm(total=len(dataloader)) as pbar:
        idx = -1

        txn = db.begin(write=True)
        for idx, (img, label) in enumerate(dataloader):
            txn = put_or_grow(txn, u'{:08}'.format(idx).encode('ascii'), dumps((img, label)))
            pbar.update()
            if (idx + 1) % write_freq == 0:
                txn.commit()
                txn = db.begin(write=True)
        txn.commit()

        keys = [u'{:08}'.format(k).encode('ascii') for k in range(idx + 1)]
        with db.begin(write=True) as txn:
            txn = put_or_grow(txn, b'__keys__', dumps(keys))

        logger.info("Flushing database ...")
        db.sync()
    db.close()


parser = argparse.ArgumentParser(description='Convert ImageNet Folder to LMDB File')
parser.add_argument('--from-dir', '-from', type=str, required=True,
    help='path to root folder. This root should contain subfolders val/ and train/')
parser.add_argument('--to-dir', '-to', type=str, required=True,
    help='path where to store the new lmdb file')
parser.add_argument('--save-as', '-save-as', choices=['raw', 'numpy'], type=str, required=True,
    help="whether to save files as jpeg or as numpy uint8-arrays. jpeg arrays "
        "take 5.6x less space and are therefore quicker to load, but need more "
        "compute to be loaded (because of an additional decoding step). Depending on "
        "the architecture, you may prefer one over the other")
parser.add_argument('--split', '-split', default=None, choices=['val', 'train'], type=str,
    help="'val' or 'train'. Defaults to None, in which case both will be made")
parser.add_argument('--workers', '-workers', default=0, type=int, 
    help="Number of workers used to load the images. Rule of thumb: 1-2 per cpu core. "
        "If 0, loading happens in main process; if > 0, then it happens in a "
        "parallel process. Default: 0")

if __name__ == '__main__':
    args = parser.parse_args()
    from_train_dir = os.path.join(args.from_dir, 'train')
    from_val_dir = os.path.join(args.from_dir, 'val')
    if args.split in ['val', None]:
        create_lmdb(
            from_val_dir, args.to_dir, 'val', args.save_as, args.workers)
    if args.split in ['train', None]:
        create_lmdb(
            from_train_dir, args.to_dir, 'train', args.save_as, args.workers)
