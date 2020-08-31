# DBPI-BlindSR
This code is official implementation of our paper. Please cite the below paper if this code is helpful.

Jonghee Kim, Chanho Jung, Changick Kim, "Dual Back-Projection-Based Internal Learning for Blind Super-Resolution," IEEE Signal Processing Letters, vol. 27, pp. 1190-1194, Jun. 2020.

This code is based on KernelGAN (https://github.com/sefibk/KernelGAN)

## Prerequisites
```
Pytorch == 1.3.0
torchvision == 0.4.1
numpy == 1.17.2
scipy == 1.3.1
tqdm == 4.36.1
Pillow == 6.2.0
```

## Installing
```
git clone github.com/prote376/DBPI-BlindSR
```

## Running the tests
```
# for X2 SR
python train.py -i test_images/ -o Results/

# for X4 SR
python train.py -i test_images/ -o Results/ --X4
```

## Examples
![Supplementary](https://user-images.githubusercontent.com/10805291/79537176-b0677a80-80bc-11ea-89cc-cad166e04eaa.jpg)
