_The project took place in Feb - Mar 2024_
# Enhanced Convolutional Occupancy Networks

~~

## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `conv_onet` using
```
conda env create -f environment.yaml
conda activate conv_onet
```
**Note**: you might need to install **torch-scatter** mannually following [the official instruction](https://github.com/rusty1s/pytorch_scatter#pytorch-140):
```
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
```

Next, compile the extension modules.
You can do this via
```
python setup.py build_ext --inplace
```

## Dataset

In this paper, we only consider ShapeNet dataset:

You can download the dataset (73.4 GB) by running the [script](https://github.com/autonomousvision/occupancy_networks#preprocessed-data) from Occupancy Networks. After, you should have the dataset in `data/ShapeNet` folder.

## Train



## View Training Results
Finally, to train a new network from scratch, run:
```
python train.py CONFIG.yaml
```
For available training options, please take a look at `configs/default.yaml`.


## Further Information
This project is based on the work of [[ECCV 2020] Peng et al. - Convolutional Occupancy Networks](https://arxiv.org/abs/2003.04618)

Please also check out the following concurrent works that either tackle similar problems or share similar ideas:
- [[CVPR 2020] Jiang et al. - Local Implicit Grid Representations for 3D Scenes](https://arxiv.org/abs/2003.08981)
- [[CVPR 2020] Chibane et al. Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion](https://arxiv.org/abs/2003.01456)
- [[ECCV 2020] Chabra et al. - Deep Local Shapes: Learning Local SDF Priors for Detailed 3D Reconstruction](https://arxiv.org/abs/2003.10983)
