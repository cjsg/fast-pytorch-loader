import os
import glob
from pathlib import Path
import numpy as np
from tensorpack.dataflow import *
from PIL import Image

class BinaryDataFlow(dataset.RNGDataFlow):
    def __init__(self, img_root_dir, name, save_as='jpeg'):
        '''
        name (str) : Must be 'train', 'val' or 'test'
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
                ValueError(
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


# print('Starting with validation set')
# ds0 = BinaryDataFlow('~/datasets/imgnet/', 'val', save_as='numpy')
# LMDBSerializer.save(ds0, os.path.expanduser('/work/cjsimon/datasets/lmdb_imgnet_uint/val.lmdb'))
# 
# print('Starting with training set')
# ds0 = BinaryDataFlow('~/datasets/imgnet/', 'train', save_as='numpy')
# LMDBSerializer.save(ds0, os.path.expanduser('/work/cjsimon/datasets/lmdb_imgnet_uint/train.lmdb'))

# ds0 = BinaryDataFlow('~/datasets/imgnet12/', 'val', save_as='jpeg')
# # ds1 = MultiProcessRunnerZMQ(ds0, num_proc=1)
# LMDBSerializer.save(ds0, os.path.expanduser('/work/cjsimon/datasets/lmdb_imgnet12/val.lmdb'))

ds0 = BinaryDataFlow('~/datasets/imgnet12/', 'train', save_as='jpeg')
ds1 = MultiProcessRunnerZMQ(ds0, num_proc=1)
LMDBSerializer.save(ds1, os.path.expanduser('work/cjsimon/datasets/lmdb_imgnet12/train.lmdb'))
