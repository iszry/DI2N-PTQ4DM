# 2023 HKU Summer Research

Code base for 2023 Summer Research Internship Programme in Department of Computer Science at **The University of Hong Kong**, assigned to the **AI, Robotics and Visual Computing**.

## Reproduce results in PTQ4DM

We found the sample distribution proposed in [PTQ4DM](https://github.com/42Shawn/PTQ4DM) does not match the experiment results. [The code](https://github.com/42Shawn/PTQ4DM/blob/main/PTQ4DM/QDrop/quant/quant_model.py) in PTQ4DM shows that they only quantize the 2D convolutional,  linear layer and the activation after these layers. However, [U-Net](https://github.com/iszry/HKU_2023_Summer_Research/blob/main/QDrop/model_structure.txt) used in [improved-diffusion](https://github.com/openai/improved-diffusion) has different structure from traditional ResNet, which indicates function from QDrop can not be applied directly. Therefore, we quantized the diffusion model again, instead of using the partly quantized model.
Then, we compared the performance of **N**ormally **D**istributed **T**ime-step **C**alibration (NDTC, proposed in PTQ4DM) and Uniformly Distributed Time-step Calibration on both ImageNet64 and CIFAR10. We found normally calibration even worse than uniform calibration, which is same to the result in [Q-Diffusion](https://github.com/Xiuyu-Li/q-diffusion).

## A further step based on PTQ4DM

We observed that IS and sFID in 8-bit PTQ4DM can reach or even exceed those in the full-precision model. However, there is always a significant loss on **FID** after quantization. Therefore, we propose some methods (D2IN) to improve the performance on FID.

| Method    |  CIFAR10   |  IS↑   |  FID↓   |  sFID↓   |   ImageNet64  |  IS↑   |  FID↓   |  sFID↓   |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|FP32|DDIM|**9.25**|**10.60**|**7.41**|DDIM|**15.20**|**19.59**|**9.45**|
|PTQ4DM|Normal|**9.38**|12.85|7.53|Normal|**15.59**|22.02|6.62|
|DI2N (Ours)|Uniform|9.25|**10.71**|**7.38**|Uniform|15.30|**19.27**|**6.63**|

> IS loss is caused by the extent of quantization. However, it still outperforms that of DDIM.

## Reference

[42Shawn/PTQ4DM: Implementation of Post-training Quantization on Diffusion Models (CVPR 2023) (github.com)](https://github.com/42Shawn/PTQ4DM)

[Xiuyu-Li/q-diffusion: [ICCV 2023] Q-Diffusion: Quantizing Diffusion Models. (github.com)](https://github.com/Xiuyu-Li/q-diffusion)

[CompVis/latent-diffusion: High-Resolution Image Synthesis with Latent Diffusion Models (github.com)](https://github.com/CompVis/latent-diffusion)

[What are Diffusion Models? | Lil'Log (lilianweng.github.io)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

[Generative Modeling by Estimating Gradients of the Data Distribution | Yang Song (yang-song.net)](https://yang-song.net/blog/2021/score/)

[openai/guided-diffusion (github.com)](https://github.com/openai/guided-diffusion)

[openai/improved-diffusion: Release for Improved Denoising Diffusion Probabilistic Models (github.com)](https://github.com/openai/improved-diffusion)

[wimh966/QDrop: The official PyTorch implementation of the ICLR2022 paper, QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization (github.com)](https://github.com/wimh966/QDrop)

[Generative Modeling by Estimating Gradients of the Data Distribution | Yang Song (yang-song.net)](https://yang-song.net/blog/2021/score/)

[CompVis/stable-diffusion: A latent text-to-image diffusion model (github.com)](https://github.com/CompVis/stable-diffusion)

[yhhhli/BRECQ: Pytorch implementation of BRECQ, ICLR 2021 (github.com)](https://github.com/yhhhli/BRECQ)

[[2304.08818] Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models (arxiv.org)](https://arxiv.org/abs/2304.08818)

[[2305.10657] PTQD: Accurate Post-Training Quantization for Diffusion Models (arxiv.org)](https://arxiv.org/abs/2305.10657)

[[1911.07190] Loss Aware Post-training Quantization (arxiv.org)](https://arxiv.org/abs/1911.07190)

[[2006.11239] Denoising Diffusion Probabilistic Models (arxiv.org)](https://arxiv.org/abs/2006.11239)

[[2102.09672] Improved Denoising Diffusion Probabilistic Models (arxiv.org)](https://arxiv.org/abs/2102.09672)

[[2010.02502] Denoising Diffusion Implicit Models (arxiv.org)](https://arxiv.org/abs/2010.02502)

[[2203.05740] QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization (arxiv.org)](https://arxiv.org/abs/2203.05740)

[[2105.05233] Diffusion Models Beat GANs on Image Synthesis (arxiv.org)](https://arxiv.org/abs/2105.05233)

[[1505.04597] U-Net: Convolutional Networks for Biomedical Image Segmentation (arxiv.org)](https://arxiv.org/abs/1505.04597)

[[2302.04304\] Q-Diffusion: Quantizing Diffusion Models (arxiv.org)](https://arxiv.org/abs/2302.04304)
