There are two steps for evaluating pretrained ZeroVL model.
1. Preparing image classification data (e.g., ImageNet) and downloading pre-trained checkpoints.
2. Modifying experiment configurations for your environment.

### 1. Preparing Training Data & Checkpoints

#### 1.1. Data Structure
The training and validation data should be placed in the "data" folder, and the expected structure is as followed:
```
CLS-LOC
|
└───train
│   └───n01440764
│   |   │   n01440764_10026.JPEG
│   |   │   ...
|   |
|   └───n01739381 
│       │   n01739381_1212.JPEG
│       │   ...
│   
└───val
│   └───n01440764
│   |   │   ILSVRC2012_val_00000293.JPEG
│   |   │   ...
|   |
|   └───n01739381 
│       │   ILSVRC2012_val_00001108.JPEG
│       │   ...
```
If the data is placed in other folders, you should revise the 'data.data_path' in yaml configs.


### 2. Linear Probing
If checkpints and datasets are ready, you could run the following scripts for linear probing:
```
python3 launch.py --task=linear_prob \
--cfg=configs/linear_prob/imagenet.yaml \
ckpt.external_resume=${path_to_ckpt} \
data.data_path=${path_to_data}
```
