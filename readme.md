# README

## Overview

The experiments are conducted on the CIFAR-10 SVHN and ImageNet datasets. Here, we provide the guidline for CIFAR-10.


## Pretrained Weight

The weights include both the diffusion model's weights and the classifier's weights. And We use the same weight with [https://github.com/NVlabs/DiffPure](DiffPure). Put the diffusion model weights into the path ```./pretrained/guided_diffusion/```


## Hyperparameters
```
strength_a              hyperparameter t_a in the paper
strength_b              hyperparameters t_b in the paper
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

Below is an example of how to run an experiment for evaluation using PGD+EOT $l_{\infty}$ attack:

```bash
python  main.py --dataset cifar10 \
    --strength_a 0.2 \
    --strength_b 0.1 \
    --threshold 0.85 \
    --attack_ddim_steps 10 \
    --defense_ddim_steps 200 \
    --forward_noise_steps 10 \
    --attack_method pgd\
    --n_iter 200 \
    --eot 20 \
    --use_cuda True \
    --port 1234
```

Below is an example of how to run an experiment for evaluation using PGD+EOT $l_{2}$ attack:

```bash
python  main.py --dataset cifar10 \
    --batch_size=64\
    --strength_a 0.2 \
    --strength_b 0.1 \
    --threshold 0.85 \
    --attack_ddim_steps 10 \
    --defense_ddim_steps 200 \
    --forward_noise_steps 10 \
    --attack_method pgd_l2\
    --n_iter 200 \
    --eot 20 \
    --use_cuda True \
    --port 1234
```

After evaluation, the original images will be stored in ```./original```, the adversarial images will be stored in ```./adv``` and the purified images will be stored in ```./pure_images```