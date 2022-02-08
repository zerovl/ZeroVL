There are two steps for evaluating pretrained ZeroVL model.
1. Preparing eval data and downloading pre-trained checkpoints.
2. Modifying experiment configurations for your environment.

### 1. Preparing Evaluation Data & Checkpoints

#### 1.1. Data Structure
The training and validation data should be placed in the "data" folder, and the expected structure is as followed:
```
data
|
└───f30k
│   │   valid_anno.csv 
|   └───valid 
│       │   valid_image_1.jpg
│       │   valid_image_2.jpg
│       │   ...
│   
└───coco
│   │   valid_anno.csv
|   └───valid (optional)
│       │   valid_image_1.jpg
│       │   valid_image_2.jpg
│       │   ...
|...
```
Note that the "valid_anno.csv" file and "valid" folder are necessary for validation datasets. If the data is placed in other folders, you should revise the 'data.data_path' in yaml configs.

#### 1.2. Annotation Structure
The "valid_anno.csv" should be provided. With reading annotation csv files with pandas, the DataFrame should be like:
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

#### 1.3. Checkpoints
Download checkpoints to the 'ckpts' folder:
```
ckpts
└───zerovl_14m.pth
└───zerovl_100m.pth
```

### 2. Evaluation
If checkpints, MSCOCO and Flickr30K datasets are ready, you could run the following scripts for evaluating zero-shot image-text retrieval on 'COCO and F30K' with 14M data pre-trained model:
```
python3 -m torch.distributed.launch \
--nproc_per_node=4 \ 
tools/retrieval_evaluation.py \
--ckpt_path=${path_to_14m_ckpt} \
--cfg=configs/clip/zerovl_14m_eval.yaml
```

For evaluation with 100M data pre-trained model:
```
python3 -m torch.distributed.launch \
--nproc_per_node=4 \ 
tools/retrieval_evaluation.py \
--ckpt_path=${path_to_100m_ckpt} \
--cfg=configs/clip/zerovl_100m_eval.yaml
```

We set the number of GPUs to 4 for fast evaluation, and it could be revised if necessary. If you want to evaluate on other datasets, modify the 'data.valid_name' in yaml file accordingly.