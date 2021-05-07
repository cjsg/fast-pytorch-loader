import os, sys
from tqdm import tqdm

from dataflow import LMDBData
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import IMG_SIZE, BATCH_SIZE, NUM_WORKERS, N_EPOCHS, WARMUP_STEPS, tqdm_args
from loaders import LMDBDataset


if __name__ == '__main__':

    # SPEED TEST

    DEFAULT_ROOT = os.path.expanduser('/work/cjsimon/datasets/lmdb_imgnet12_uint/')
    FILE_PATH = os.path.join(DEFAULT_ROOT, 'train.lmdb')

    print(f'\nworkers={NUM_WORKERS}'
          f'\nfilepath={FILE_PATH}')

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
                # img = img.cuda()
                # lab = lab.cuda()
                pbar.update()
                if ix == (len(loader)-1):
                    break
