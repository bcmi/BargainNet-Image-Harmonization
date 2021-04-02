# BargainNet
This repository contains the official PyTorch implementation of the following paper:

> **BargainNet: Background-Guided Domain Translation for Image Harmonization**<br>
>
> [Wenyan Cong](https://wenyancong.com/), [Li Niu](http://bcmi.sjtu.edu.cn/home/niuli/), [Jianfu Zhang](http://scholar.google.com/citations?user=jSiStc4AAAAJ&hl=zh-CN), Jing Liang, Liqing Zhang<br>MoE Key Lab of Artificial Intelligence, Shanghai Jiao Tong University<br>
> https://arxiv.org/abs/2009.09169<br>Accepted by **ICME2021** as **Oral**.





Our trained model can be found in [Baidu Cloud](https://pan.baidu.com/s/1E9Dj_DeRCLiZgVp625SjLA) (access code: 5jz4). Download and put it under this directory. To test and re-produce the reported results in our paper, run:

`python test.py  --name bargainnet  --model bargainnet --dataset_mode iharmony4 --is_train 0  --norm batch --preprocess resize --gpu_ids 0  --input_nc 20 --netG s2ad`

## Bibtex
If you find this work is useful for your research, please cite our paper using the following **BibTeX  [[pdf]()] [[supp]()] [[arxiv](https://arxiv.org/abs/2009.09169)]:**

```
@inproceedings{BargainNet2021,
title={{BargainNet}: Background-Guided Domain Translation for Image Harmonization},
author={Wenyan Cong and Li Niu and Jianfu Zhang and Jing Liang and Liqing Zhang},
booktitle={ICME},
year={2021}}
```

