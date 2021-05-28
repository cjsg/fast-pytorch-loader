from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'LMDB-based dataset(-loader) for PyTorch'
LONG_DESCRIPTION =\
"""
    Package to create .lmdb files from image directories (arranged according to
    PyTorch's standart ImageDataset) and to load them in a PyTorch Dataset
"""

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="lmdbdataset", 
        version=VERSION,
        author="Carl-Johann Simon-Gabriel",
        author_email="<johann2706@hotmail.com>",
        url="https://github.com/cjsg/fast-pytorch-loader",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['argparse', 'numpy', 'Pillow', 'lmdb'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['pytorch', 'fast image loader', 'lmdb'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education / Research",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Linux :: Linux",
            "Operating System :: Microsoft :: Windows",
        ]
)
