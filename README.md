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
