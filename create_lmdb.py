import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import argparse
from tensorpack.dataflow import *  # dataflow package alone is not enough

# import argparse

class BinaryDataFlow(dataset.RNGDataFlow):
    def __init__(self, img_root_dir, name, save_as='jpeg'):
        '''
        name (str) : Must be 'train' or 'val'
        save_as (str) : 
            'jpeg' (for binary-compressed RGB jpegs; or 'numpy' for uint8 numpy arrays
        '''
        assert name in ['train', 'val', 'test']
        img_root_dir = os.path.expanduser(img_root_dir)
        self.root = os.path.join(img_root_dir, name)
        self.save_as = save_as
        self.img_file_names, self.labels = self.get_img_file_names(self.root)
        
    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        for ix, (fname, label) in enumerate(zip(self.img_file_names, self.labels)):
            # if ix % 100 == 0:
            #     print(f'{ix:05d}, {label:02d}, {fname}')
            if self.save_as == 'jpeg':
                with open(fname, 'rb') as f:
                    jpeg = f.read()
                jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
                yield [jpeg, label]
            elif self.save_as == 'numpy':
                image = Image.open(fname).convert('RGB')
                image = np.array(image, np.uint8).transpose((2, 0, 1))
                yield [image, label]
            else:
                raise ValueError(
                    f"save_as must be 'jpeg' or 'numpy'.  Given {self.save_as}")

    def get_img_file_names(self, root):
        print('Getting images...')
        assert os.path.isdir(root), f'No directory {root}'
        image_ps = sorted(glob.glob(root+'/**/*.JPEG', recursive=True))
        image_ps = [Path(image_p) for image_p in image_ps]
        assert len(image_ps) > 0, f'No images found in {root}'
        print('Found {} images'.format(len(image_ps)))

        subdirs= set([self.parent(image_p) for image_p in image_ps])
        subdir_to_label = {subdir: lab for lab, subdir in enumerate(subdirs)}
        labels = [subdir_to_label[self.parent(image_p)] for image_p in image_ps]

        return image_ps, labels

    def parent(self, path):
        # find parent of path
        path = os.path.expanduser(path)
        return os.path.abspath(path+'/..')


parser = argparse.ArgumentParser(description='Convert ImageNet Folder to LMDB File')
parser.add_argument('--from-dir', '-from', type=str,
    help='path to ImageNet root folder. This root should contain subfolders val/ and train/')
parser.add_argument('--to-dir', '-to', type=str, help='path where to store the new lmdb file')
parser.add_argument('--save-as', '-save-as', choices=['numpy', 'jpeg'], type=str, required=True,
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

    splits = ['val', 'train'] if args.split is None else [args.split]
    from_dir = os.path.expanduser(args.from_dir)
    to_dir = os.path.expanduser(args.to_dir)

    assert os.path.isdir(from_dir)
    assert os.path.isdir(to_dir)

    for split in splits:
        split_str = 'validation' if split == 'val' else 'training'
        to_file = os.path.join(to_dir, split+'.lmdb')

        print(f'Starting to encode the {split_str} set as a {args.save_as}-lmdb file')
        ds = BinaryDataFlow(from_dir, split, save_as=args.save_as)
        if args.workers > 0:
            ds = MultiProcessRunnerZMQ(ds, num_proc=args.workers)
        LMDBSerializer.save(ds, to_file)
