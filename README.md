## Dataset: ModelNet10 
http://modelnet.cs.princeton.edu/#

## Data preparation 
https://github.com/guoguo12/modelnet-cnn3d_bn

.off -(1.voxelize)-> .binvox -(2.read)-> Numpy array

1. Voxelize\
Read in [.off] file and change it into [.binvox] file of binary voxel data , using [binvox](http://www.patrickmin.com/binvox/) program.

2. Create Numpy array\
Read [.binvox] file into Numpy array(30,30,30), using [binvox_rw.py](https://github.com/Sooram/ModelNet-3dCNN-tensorflow/blob/master/binvox_rw.py) and [prepare_data.py](https://github.com/Sooram/ModelNet-3dCNN-tensorflow/blob/master/prepare_data.py).

## Network architecture
very simple: ccp-ccp-output
![Overview](https://github.com/Sooram/ModelNet-3dCNN-tensorflow/blob/master/network.PNG)
