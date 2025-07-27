# MSAGRegNet
## Related information
<!--Here is the code of "VRNet: Learning the Rectified Virtual Corresponding Points for 3D Point Cloud Registration" (``https://ieeexplore.ieee.org/abstract/document/9681904``), which proposes to rectify the virtual corresponding points to avoid the degeneration problem.-->
Here is the MSAGRegNet code.

<!--Note: the code is being prepared. -->

## Implementation
The code is tested with Pytorch 1.6.0 with CUDA 10.2.89. Prerequisites include scipy, h5py, tqdm, etc. Your can install them by yourself.

The ModelNet40 dataset can be download from:
```
https://github.com/WangYueFt/dcp
```
and put it to data/ModelNet/

Start training with the command:
```
python train_modelnet40.py
```

Start testing with the command:
```
python test_modelnet40.py
```

## Acknowledgement
The code is insipred by DCP, PRNet, RPMNet and VRNet, etc.
