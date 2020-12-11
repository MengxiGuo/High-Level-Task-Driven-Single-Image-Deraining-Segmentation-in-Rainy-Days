# High-Level-Task-Driven-Single-Image-Deraining-Segmentation-in-Rainy-Days
![avatar](https://github.com/MengxiGuo/High-Level-Task-Driven-Single-Image-Deraining-Segmentation-in-Rainy-Days/blob/main/48a4d4f527c019afcdcf7a190274c9a.png)

# Our model
![avatar](https://github.com/MengxiGuo/High-Level-Task-Driven-Single-Image-Deraining-Segmentation-in-Rainy-Days/blob/main/43f84a0d0aeb1a56343066f71065940.png)

# Results
![avatar](https://github.com/MengxiGuo/High-Level-Task-Driven-Single-Image-Deraining-Segmentation-in-Rainy-Days/blob/main/3b244b3817c35190af365946f008a43.png)

# additional experiment 
## Combination with the state-of-art Deraining Model
The traditional OJT approach makes the high-level task driven model independently, and it does not well combine with many excellent deraining work. Thanks to our special design, our SRRN and TJT training methods can be well combined with the existing deraining models. We combine our SRRN with SOTA deraining model DuRN(DuRN + R), and use OJT and TJT for training on our Raindrop-Cityscapes datasets respectively. The results in Table2 show that after combining our SRRN, the performance of the model improves on PSNR/SSIM and mIoU. This proves the feasibility of combining our SRRN model with the SOTA model. However, compared with the DuRN + R + OJT, the improvement of DuRN + R + TJT is more significant. This further indicates that in addition to the improvement brought by the network structure, the optimization of joint training method also further improves the performance.

![avatar](https://github.com/MengxiGuo/High-Level-Task-Driven-Single-Image-Deraining-Segmentation-in-Rainy-Days/blob/main/aedd3baa67f923fa7e821bc74adf642.png)

# Raindrop-Cityscapes Dataset: 
https://drive.google.com/file/d/1KyBKJ-gWQ1hEstfzKGmtcQs0wfDfqsxi/view?usp=sharing

# Demo Vedio
http://www.bilibili.com/video/BV1da411F7fN?share_medium=android&share_source=copy_link&bbid=21F89BD0-1545-41A1-A7D9-9D3FED097E2A30816infoc&ts=1607689316096

# Paper link
https://link.springer.com/chapter/10.1007/978-3-030-63830-6_30

# Supplementary
https://drive.google.com/file/d/1yL10cyBL_DeHhd9bJc_jI_GQNPAXdEYq/view?usp=sharing

# Citation
@inproceedings{guo2020high,
  title={High-Level Task-Driven Single Image Deraining: Segmentation in Rainy Days},
  author={Guo, Mengxi and Chen, Mingtao and Ma, Cong and Li, Yuan and Li, Xianfeng and Xie, Xiaodong},
  booktitle={International Conference on Neural Information Processing},
  pages={350--362},
  year={2020},
  organization={Springer}
}
