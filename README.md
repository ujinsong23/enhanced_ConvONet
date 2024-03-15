_The project took place in Feb - Mar 2024_

_Project for EE367(Computational Imaging), Stanford University_
# Enhanced Convolutional Occupancy Networks

The Enhanced Convolutional Occupancy Network aims to create a more effective representation of 3D scenes by
incorporating all three coordinate values of 3D coordinates in its plane encoding process. In contrast to the Convolutional Occupancy
Network(ConvONet), which discards non-projected coordinates during orthographic projection onto canonical planes for 2D plane
encoder, the proposed method integrates these additional dimensions to improve feature extraction. Experimental results showcase
the method’s effectiveness in achieving faster convergence and generating higher-quality 3D reconstructions, as evidenced by higher
Intersection over Union (IoU) scores, while maintaining a comparable computational burden in terms of time and memory usage when
compared to existing methods.
![alt text](https://github.com/ujinsong23/enhanced_ConvONet/blob/master/Training%20Pipeline.jpg)


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
To train model, use the following command
```
python train.py --encoder-type ENCODER
```

Replace ENCODER with one of the following options: "sum", "range", "mix", or "none". 
This command will automatically load the corresponding configuration files from the `configs/pointcloud` folder.

```
shapenet_3plane_indoor_weightmix.yaml
shapenet_3plane_indoor_weightnone.yaml
shapenet_3plane_indoor_weightrange.yaml
shapenet_3plane_indoor_weightsum.yaml
```

The output for executing the code has already been stored in the `out` folder.


## View Training Results
Finally, to plot training error and validation iou, use the following command
```
python plot_result
```
This command will generate plots for all four different models: "sum", "range", "mix", and "none".

If you want to plot the graph for only one of these modes, use the following command:

```
python plot_result --plot-mode ENCODER
```
Replace ENCODER with one of the following options: "sum", "range", "mix", or "none". This command will generate a plot specifically for the chosen encoder type.


## Further Information
This project is based on the work of [[ECCV 2020] Peng et al. - Convolutional Occupancy Networks](https://arxiv.org/abs/2003.04618)

Please also check out the following concurrent works that either tackle similar problems or share similar ideas:
- [[CVPR 2020] Jiang et al. - Local Implicit Grid Representations for 3D Scenes](https://arxiv.org/abs/2003.08981)
- [[CVPR 2020] Chibane et al. Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion](https://arxiv.org/abs/2003.01456)
- [[ECCV 2020] Chabra et al. - Deep Local Shapes: Learning Local SDF Priors for Detailed 3D Reconstruction](https://arxiv.org/abs/2003.10983)
