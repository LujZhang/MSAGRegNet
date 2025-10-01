# MSAGRegNet
## Related information
Here is the MSAGRegNet code.

<!--Note: the code is being prepared. -->

## Implementation
The code is tested with Pytorch 1.6.0 with CUDA 10.2.89. Prerequisites include scipy, h5py, tqdm, etc. You can install them by yourself.

The BitFace3D and preprocessed BARIM dataset and pre-weight of our train can be obtained by sending an email to lujianz@bit.edu.cn from:

and put it to ./MSAGNet/, the dataser neednt any preprocesse.

Start training with the command:
```
python train_bitface3d.py
```

Start testing with the command:
```
python test_bitface3d.py
```

## Acknowledgement
The code is insipred by DCP, PRNet, RPMNet and VRNet, etc.
