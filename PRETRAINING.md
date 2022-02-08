There are two steps for training your own model with ZeroVL.
1. Preparing training data.
2. Modifying experiment configurations for your environment.

### 1. Preparing Training Data

#### 1.1. Data Structure
The training and validation data should be placed in the "data" folder, and the expected structure is as followed:
```
data
|
└───cc3m
│   │   train_anno.csv
│   │   valid_anno.csv (optional)
│   └───train
│   |   │   train_image_1.jpg
│   |   │   train_image_2.jpg
│   |   │   ...
|   |
|   └───valid (optional)
│       │   valid_image_1.jpg
│       │   valid_image_2.jpg
│       │   ...
│   
└───sbu
│   │   train_anno.csv
│   │   valid_anno.csv (optional)
│   └───train
│   |   │   train_image_1.jpg
│   |   │   train_image_2.jpg
│   |   │   ...
|   |
|   └───valid (optional)
│       │   valid_image_1.jpg
│       │   valid_image_2.jpg
│       │   ...
|...
```
Note that the "valid_anno.csv" file and "valid" folder are only necessary for validation datasets. If the data is placed in other folders, you should revise the 'data.data_path' in yaml configs.

#### 1.2. Annotation Structure
A pre-training dataset folder should contain "train_anno.csv". If a dataset is used for both training and validation, the "valid_anno.csv" should be provided. With reading annotation csv files with pandas, the DataFrame should be like:
```
           image     image_id                                            caption  caption_id
0       289333.jpg    289333        A plate filled with pasta covered in sauce.       27429
1       220152.jpg    220152        Two skiers prepare to travel down the slope      167876
2       513351.jpg    513351    A computer mouse and pad with a cup of coffee.       225618
3       366379.jpg    366379  Snowboarder on top of his board overtop of snow.       763508
4       366599.jpg    366599  A skate boarder practicing his tricks on the r...      721012
...            ...       ...                                                ...         ...
585304  462037.jpg    462037  Two girls playing video games using a wireless...       40920
585305  122335.jpg    122335             A young man standing in front of a TV.      205181
585306  254850.jpg    254850  Man leans his head forward while holding up Wi...       47085
585307  451803.jpg    451803   A keyboard, mouse, and mouse pad for a computer.      225270
585308  576212.jpg    576212  Two keyboards on a table connected to a comput...      237806

[585309 rows x 4 columns]

'image'[str]: image filenames.
'image_id'[int]: image ids for validation.
'caption'[str]: text annotations.
'caption_id'[int]: text ids for validation.
```

Note that an image might be coupled with N texts, and we repeatly save the image filenames and ids for N times for simple usage.


### 2. Training & Validation
We use the PyYaml to modify configurations for better readability and interaction. 
#### 2.1. TL;DR
Considering that SBU, CC3M, MSCOCO and Flickr30K datasets are relatively easy to obatin, we choose these image-text datasets for pre-training and validation in our default config file. If there are 8 V100 GPUs and the pre-training datasets (SBU and CC3M) and validation datasets (MSCOCO and Flickr30K) are ready, you could run the following scripts for pre-training on 'SBU and CC3M' and zero-shot validation on 'COCO and F30K':
```
python3 launch.py --task=clip --nproc_per_node=8 --exp_name=your_exp_name --cfg=./configs/clip/zerovl_dga.yaml
```
This configuration enables all heuristics mentioned in the paper. Otherwise, you should read the following parts and revise the experiment configuration according to your requirements.

#### 2.2. Modifying Configurations
##### 2.2.1. Fundamental settings
   ```
    ckpt:
        dir: ./output                   # the path for saving ckpts
        step_interval: 2000             # save the model every 2000 steps for auto resume
    
    data:
        exp_name: your_exp_name # experiment folder name for saving ckpts

        train_name: [your_dataset_1, your_dataset_2, ...] 
        valid_name: [your_dataset_3, your_dataset_4, ...]

        batch_size: 1024        # the total training batch size on all devices.
        batch_size_val: 1024

    transforms:
        train_transforms: [random_resize_crop, autoaug] # for training images
        valid_transforms: [resize, center_crop]         # for validation images
        resize:                                         # the specific parameter for resize
            size: 256 
        center_crop:                                    # the specific parameter for center_crop
            size: 224 
        random_resize_crop:                             # the specific parameter for random_resize_crop
            size: 224 
            scale: [0.6, 1.0]

    model:
        image_encoder:
            name: timm_modelzoo             # use any model provided in timm
            tag: vit_base_patch16_224_in21k # use ViT-B/16 pre-trained on ImageNet-21K
            embedding_dim: 768              # the dim of model's final output
            pretrained: True
            trainable: True
        text_encoder:
            name: huggingface_modelzoo      # use any model provided in huggingface
            tag: bert-base-uncased          # use Bert-Base provided by huggingface
            embedding_dim: 768              # the dim of model's final output
            pretrained: True
            trainable: True
        projection:
            dim: 512                        # 512-dim for calculating the NCE loss

    loss:
        name: NCE 

    optim:
        name: torch.optim.AdamW
        param: 
            betas: !!python/tuple [0.9, 0.98]
            eps: 1.0e-6
            weight_decay: 0.001
        lr:
            name: cosine_schedule_with_warmup_min_lr_scale
            init: 1.0e-4
            warmup_proportion: 0.025
            param:
                num_cycles: 0.5
                min_lr_scale: 0.1
   ```

##### 2.2.2. Incremental settings
The following are essential components proposed in our paper. 
###### 1. Debiased Sampling
   ```
    data:
        train_type: debias # choose from [shuffle, sequential, debias]
   ```
###### 2. Coin flipping mixup
   ```
    loss:
        name: MixUpNCE  # MixUpNCE will enable the coin flipping mixup
        mixup:
            beta: 0.1   # the beta for mixup calculation.
   ```
If you want to disable coin flipping mixup, you should change loss.name to NCE.
###### 3. Decoupled gradient accumulation (DGA).
   ```
    runner:
        name: clip_bsgs         # we implement an independent runner for DGA
        stable_random: step     # for stable training

    data:
        batch_size: 16384       # the effective batchsize can be increased to 16384
        batch_size_train: 1024  # batch size for each sub-iteration
        batch_size_val: 1024
    
   ```
If you want to disable DGA, you should change data.batch_size to a small value (e.g., 1024 and 512).
###### 4. Gather backward.
   ```
    loss:
        nce_loss:
            gather_backward: True # False will stop the gradient of gathered embeddings
   ```

### 3. Logging with Wandb
It is highly recommended to manage your experiments with [Wandb](https://wandb.ai/site). Wandb can be regarded as a cloud version tensorboard. Similar to tensorboard, each experiment and its yaml config will be automatically logged by this library. Besides, Wandb will automatically draw various curves for the experiment.

After installing and configuring Wandb, you need to revise the following parameters:
```
wandb:
    enable: True
    project: your_proj
    entity: your_entity
```
Then, our codebase will automatically upload training logs to your Wandb workspace.