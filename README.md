<img src="./docs/logo.png" width="150" >

This repository contains source code necessary to reproduce the results presented in the paper [ZeroVL: A Strong Baseline for Aligning Vision-Language Representations with Limited Resources](https://arxiv.org/abs/2112.09331).

Pioneering dual-encoder pre-training works (e.g., CLIP and ALIGN) require a tremendous amount of data and computational resources (e.g., billion-level web data and hundreds of GPUs), which prevent researchers with limited resources from reproduction and further exploration. 
To this end, we provide a comprehensive training guidance, which allows us to conduct dual-encoder multi-modal representation alignment with limited resources. Meanwhile, we provide a reproducible strong baseline of competitive results, namely **ZeroVL**, with publicly accessible academic datasets and a popular experimental environment. 

## Performance
Image-text retreival **RSUM** scores on MSCOCO and Flickr30K datasets:
method  | computation | data  | COCO(zs.) | COCO(ft.) | F30K(zs.) | F30K(ft.) |
--------| :--------:  | :---: | :-------: | :-------: | :-------: | :-------: |
CLIP	  | 256 V100    | 400M  | 400.2     | -         | 540.6     | -         |
ALIGN	  | 1024 TPUv3  | 1800M | 425.3     | 500.4     | **553.3** | **576.0** |
baseline| 8 V100      | 14.2M | 363.5     | 471.9     | 476.8     | 553.0     |
ZeroVL	| 8 V100      | 14.2M | 425.0     | 485.0     | 536.2     | 561.6     |
ZeroVL	| 8 V100      | 100M  | **442.1** | **500.5** | 546.5     | 573.6     |
zs.: zero-shot setting, ft.: fine-tuned setting.

## Installation
Requirements:
- Python 3.7
- Pytorch 1.8.1
- torchvision 0.9.1
- cuda 11.1
  
Install requirements:
```
pip install -r requirements.txt
```

## Getting Started
Check [GETTING_STARTED.md](GETTING_STARTED.md) for codebase usage.

## Model Zoo
We will release pre-trained models soon.

## Citing ZeroVL
If you use ZeroVL in your research or wish to refer to the baseline results, please use the following BibTeX entry.
```BibTeX
@article{cui2021zerovl,
  title={ZeroVL: A Strong Baseline for Aligning Vision-Language Representations with Limited Resources},
  author={Cui, Quan and Zhou, Boyan and Guo, Yu and Yin, Weidong and Wu, Hao and Yoshie, Osamu},
  journal={arXiv preprint arXiv:2112.09331},
  year={2021}
}
```

## License
ZeroVL is released under the MIT license. See [LICENSE](LICENSE) for details. 