# README

## Overview

The experiments are conducted on the CIFAR-10 SVHN and ImageNet datasets. Here, we provide the guidline for CIFAR-10.


## Pretrained Weight

The weights include both the diffusion model’s weights and the classifier’s weights. We use the same weights as [https://github.com/NVlabs/DiffPure](DiffPure). \
Place the diffusion model weights for CIFAR10, named ```checkpoint_8.pth```, and for ImageNet, named  ```256x256_diffusion_uncond.pt```, in the path ```./pretrained/guided_diffusion/```. \
The classifier weights for WideResNet28-10 (CIFAR10) will be automatically downloaded into ```./models/cifar10/L2/Standard.pt```. For the classifier weights of WideResNet70-16 (CIFAR10), you need to download them manually to  ```./pretrained/``` and load them according to the instructions in [https://github.com/NVlabs/DiffPure](DiffPure).

## DataSet

All datasets should be placed in the path ```./datasets/{}```. CIFAR10 and SVHN will be automatically downloaded by torchvision. For ImageNet, you need to download it manually. Unlike DiffPure, we do not use LMDB.

## Hyperparameters
```
strength_s              hyperparameter t_s in the paper
strength_l              hyperparameters t_l in the paper
threshold               default,value-based,hyperparameter tau in the paper
threshold_percent       percent-based
attack_ddim_steps       surrogate process
forward_noise_steps     hyperparameter U in the paper
num_ensemble_runs       The number of ensemble runs for purification in defense
n_iter                  The nubmer of iterations for the attack generation
eot                     The number of EOT samples for the attack
```


## Running Experiments

First, install the required environment:
```bash
pip install -r requirements.txt
```

### Time Estimation

On GPU 4090 with 24 GB memory, the experiment on cifar10 with the classifier of WideResNet-28-10 will cost about 4 hours.

### Example Evaluation

Below is an example of how to run an experiment on CIFAR10:

```bash
sh scripts/cifar10.sh
```

Below is an example of how to run an experiment on imagenet:

```bash
sh scripts/imagenet.sh
```

# Visualization
After evaluation, the original images will be stored in ```./original```, the adversarial images will be stored in ```./adv``` and the purified images will be stored in ```./pure_images```.