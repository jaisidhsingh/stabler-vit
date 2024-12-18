# Improving ViT with nGPT + differential attention

Tries to answer the following questions:
1. is <a href="https://arxiv.org/pdf/2410.01131">nGPT</a> (learning on hyper-spheres) beneficial for ViTs?
2. does <a href="https://arxiv.org/pdf/2410.05258">differential attention</a> reduce noise in ViTs?
3. is there a net positive effect when 1. and 2. are combined?

*Note*: I used some old ViT modelling code to quickly set up this experiment. After testing this, I'd be open to PRs which release these models using the modelling from <a href="https://github.com/huggingface/pytorch-image-models">`timm`</a> for easier access in mainstream model hubs.


## Features/Todo list
- [x] Vanilla ViT components
- [x] Differential Attention Layer
- [x] Vanilla ViT modelling
- [x] DiffAttnViT modelling
- [ ] nViT modelling
- [ ] nDiffAttnViT modelling
- [ ] Test forward pass
- [ ] Setup ImageNet-1k dataset
- [ ] Training code
- [ ] Wandb integration
- [ ] Set original hyper-parameters
- [ ] Run training script
- [ ] Add results to README


## Requirements
```
torch
einops
torchvision
```