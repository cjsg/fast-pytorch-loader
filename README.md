# Fast LMDB-file based image(net) loaders for PyTorch

This repo provides fast pytorch-friendly loaders for ImageNet, based on LMDB
files. It is based on the tensorpack.dataflow package and inspired by
[this](https://github.com/AnjieCheng/Fast-ImageNet-Dataloader) github repo.
But the solution proposed ended up being closer to
the one from the
[Efficient-PyTorch](https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py)
repo, which I discovered only later.

## Requirements

- `lmdb`
- `tensorpack` (but tensorflow is not needed)
- `torch`
- Others? -> test!

## Usage

The idea behind this loader is to put all (ImageNet) images into 1 big `.lmdb`
file, which dramatically reduces access overheads (see sections below).

#### Creating the LMDB data file

To create the `.lmdb` file from the ImageNet JPEG images, assuming that these
images are organized as needed for the official PyTorch ImageNet loader, use:

```python
python create_lmdb --from_dir SOURCE_DIR/ --to_dir TARGET_DIR/ --split TRAIN_OR_VAL --save-as FORMAT
```
where `SOURCE_DIR` is the path to the ImageNet root directory (which should
contain `train/` and `val/`directories; `TARGET_DIR/` is the path to the
directory where to store the `.lmdb` file(s), and `FORMAT` can take 2 values:
`jpeg` or `numpy` depending on whether you want the images encoded as JPEG
images or as numpy array inside the `.lmdb` files (see below).

#### The loader

To use the loader, load the LMDBDataset and then the usual pytorch loader.

```python
from loaders import LMDBDataset
dataset = LMDBDataset(root, split, transform, transform_target,shuffle, imgtype)
# OR dataset = LMDBIterDataset(root, split, transform, transform_target, imgtype)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
```
where `root` is the path to the dir containing the `.lmdb` (created above),
`split` is `val` or `train`, `imgtype`is `numpy` or `jpeg` (must the same than
the `FORMAT` used in `create_lmdb`) and `tranform`, `transform_target` are the
usual pytorch image/target transforms to be applied to every image.

**Beware**: `LMDBIterDataset` does not use any shuffling. Since, when using the
`LMDBIterDataset`with multiple workers, the dataset gets replicated, there will
be duplicates in the loaded data. If you want to use the iterable dataset and
have some shuffling, we recommend using
`torch.utils.data.BufferedShuffleDataset`.



## Background 

### Avoiding file opening overheads

The issue with pytorch's default ImageDataset/loader is that it opens each
image file individually. This induces 2 kinds of overheads at evevery opening:

1. The system (more precisely, the harddrive reading head) must find the
beginning of the image file, which, on HDDs, can be slow.
2. The system needs to check access permissions/rights for every new image.

To avoid these overheads, this solution dumps all imagenet images into one
file (actually 2: one for the training set, one for the validation), using an
LMDB format.

#### Iterative vs key-based data-loaders

One advantage of LMDB files is that they can be accessed either
sequentially/iteratively, or via key-value mappings (i.e., each key maps to one
image, using a dictionnary data-structure). Hence, you need to check access
rights only once (when opening the lmdb file), i.e., the overhead related to 2.
disappears. If you decide to load the data sequentially, then the overhead
related to 1. disappears as well; but then, you data does not get shuffled,
which is generally a good idea in deep learning. This solution is implemented
in `uint_lmdb_iterloader.py`. Instead, you can also access the images by the
(shuffled) keys, but then, of course, you keep the overhead related to 1. This
is implemented in `uint_lmdb_loader.py`.

_Remark_: These loaders were initially written for the cluster of the
Max-Planck-Institute for Intelligent Systems. Because of its specific cluster
architecture, the huge overhead was related to point 2, but not to 1. So,
there, I recommend using the `uint_lmdb_loader.py`.

#### JPEG vs np.array-of-uint files

We provide two options for the LMDB files: one that saves/loads images in JPEG
format, and one that saves/load numpy uint-arrays. The advantage of the JPEG
format is that it takes 5.6x less memory/space than the numpy arrays, and can
therefore also be loaded more quickly. The disadvantage is that the
de-compression of the image (i.e., the conversion to the numpy array format,
which is needed anywway) can take quite some time (because it is executed on
CPU, not GPU) and become the bottleneck. (That can of course be counter-acted
by using more CPUs.) In that case, I recommend using the numpy array version.


## TODO:

- make this package independent of tensorpack.dataflow (i.e., essentially,
  re-write the create_lmdb using directly the `lmdb` python package, instead of
  the tensorpack.dataflow wrapper). See in particular how it is done in
  [here](https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py).
