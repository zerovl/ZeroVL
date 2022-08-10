<img src="./docs/logo.svg" width="150" >

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

ImageNet-1K linear probing results:
method  | data  | backbone  | top-1     |
------  | :---: | :------:  | :---:     |
CLIP    | 400M  | ViT-B/16  | 80.2      |
ZeroVL  | 100M  | ViT-B/16  | **80.6**  |

## Installation
Requirements:
- Python 3.7
- Pytorch 1.8.1
- torchvision 0.9.1
- cuda 11.1
  
Install requirements:
```
pip3 install -r requirements.txt
```

## Pre-training
Check [PRETRAINING.md](PRETRAINING.md) for codebase usage.

## Model Zoo
ZeroVL 14M weights: [Google Drive](https://drive.google.com/file/d/1Pb5o7EJTCXJyn0vIOE1vdGnJ4l_4mfG2/view?usp=sharing), [Baidu Pan](https://pan.baidu.com/s/1D5RKc2UVhK1y4xGRdIvQeA?pwd=himv)

ZeroVL 100M weights: [Google Drive](https://drive.google.com/file/d/1tkAp3ENPsFMeaW8nbk9bu1zFO-YfhjH5/view?usp=sharing), [Baidu Pan](https://pan.baidu.com/s/1FRsYJIRdP54D6L2veaIDYw?pwd=s42h)

## Evaluation
Check [EVALUATION.md](EVALUATION.md) for codebase usage.

## Linear Probing
Check [LINEAR.md](LINEAR.md) for codebase usage.

## Citing ZeroVL
If you use ZeroVL in your research or wish to refer to the baseline results, please use the following BibTeX entry.
```BibTeX
@article{cui2022zerovl,
  title={Contrastive Vision-Language Pre-training with Limited Resources},
  author={Cui, Quan and Zhou, Boyan and Guo, Yu and Yin, Weidong and Wu, Hao and Yoshie, Osamu and Chen, Yubo},
  journal={ECCV},
  year={2022}
}
```

## License
ZeroVL is released under the MIT license. See [LICENSE](LICENSE) for details. 
