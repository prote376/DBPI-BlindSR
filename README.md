# DBPI-BlindSR

This code is official implementation of the paper we submitted. (Paper information will be updated if it is accepted.)
This code is based on KernelGAN (https://github.com/sefibk/KernelGAN)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Pytorch == 1.3.0
torchvision == 0.4.1
numpy == 1.17.2
scipy == 1.3.1
tqdm == 4.36.1
Pillow == 6.2.0
```

### Installing

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
