# Fast LMDB-file based image(net) loaders for PyTorch

This repo provides fast pytorch-friendly loaders for ImageNet, based on LMDB
files. It is based on the tensorpack.dataflow package and inspired by
[this](https://github.com/AnjieCheng/Fast-ImageNet-Dataloader) github repo.
But the solution proposed ended up being closer to
the one from the
[Efficient-PyTorch](https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py)
repo, which I discovered only later.

## Requirements

- `Pillow`, `numpy`, `argparse`, `pathlib`
- `lmdb`
- `tensorpack` (but tensorflow is not needed)

**Remark**: Currently, when loading the `lmdbdataset` package, the underlying
`tensorpack` package will try to load tensorflow and print `Failed to import
tensorflow` if it failed. You can ignore this message.

## Installation

To install the package:
```
    git clone https://github.com/cjsg/fast-pytorch-loader.git
    cd fast-pytorch-loader/
    python setup.py install
```
To install it in development mode, replace `python setup.py install` by `python
setup.py build develop`.


## Usage

The idea behind this loader is to put all (ImageNet) images into 1 big `.lmdb`
file, which dramatically reduces access overheads (see section Background below).

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

#### The datasets and loaders

This package implements two LMDB-based loading proceedures: one on the iterable
`LMDBIterDataset` combined with a custom `BufferedDataLoader`,
```python
from loaders import LMDBIterDataset, BufferedDataLoader
dataset = LMDBIterDataset(root, split, transform, transform_target, imgtype)
trainloader = BufferedDataLoader(buffer_size, dataset, batch_size=bs,
    persistent_buffer, num_workers, **pytorch_loader_kwargs)
```
and one based on the map-style `LMDBDataset` combined with the usual PyTorch `DataLoader`:
```python
from loaders import LMDBDataset
dataset = LMDBDataset(root, split, transform, transform_target, shuffle, imgtype)
loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers)
```
Here, `root` is the path to the dir containing the `.lmdb` (created above),
`split` is `val` or `train`, `imgtype`is `numpy` or `jpeg` (must the same than
the `FORMAT` used in `create_lmdb`) and `tranform`, `transform_target` are the
usual pytorch image/target transforms to be applied to every image.

**On the MPI-cluster, and more generally on distributed file systems and/or
with non-SSD hard-drives, the `LMDBIterDataset` solution will be faster,**
because it reads the data in the order it has been stored (see section
Background). Our custom `BufferedDataLoader` is then used to shuffle the data.
It wraps and sub-classes PyTorch's usual `DataLoader` to keep a buffer of
images. It adds every new image from the `LMDBIterDataset` to its internal
buffer, and then randomly samples an image from the buffer.

_Remark_: PyTorch 1.8.1 also provides `torch.utils.data.BufferedShuffleDataset`
which can be used to wrap any iterable dataset and then passed to the usual
PyTorch `DataLoader`. However, with this solution, every worker has its own
buffer, whereas with our `BufferedDataLoader`, the buffer is centralized
accross workers. This is important in our case, since, with `LMDBIterDataset`,
the dataset gets partitioned between workers (i.e., each worker sees a
different chunk).


## Background 

### Avoiding file opening overheads: iterative vs map-style datasets

The issue with PyTorch's default ImageDataset/loader is that it opens each
image file individually. This induces 2 kinds of overheads at evevery opening:

1. The system needs to check access permissions/rights for every new image.
2. The system (actually, the harddrive reading head) must move to the
beginning of the image file, which, on HDDs, can be slow.

By putting all files into one LMDB file, we avoid the overhead related to 1.
By reading them in sequence (i.e., in the order on the disk), we avoid the
overhead 2. Said differently, `LMDBDataset` avoids overhead 1.,
`LMDBIterDataset` avoids both.

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
