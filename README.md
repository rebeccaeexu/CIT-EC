# CNN Injected Transformer for Exposure Correction



## Introduction

This repository is the **official implementation** of the paper, "CNN Injected Transformer for Exposure Correction".



### Environment

* basicsr==1.4.2
* scikit-image==0.15.0



### Dataset Preparation

* Download datasets

1.  MSEC dataset (please refer to https://github.com/mahmoudnafifi/Exposure_Correction)

2.  SICE dataset (please refer to https://github.com/KevinJ-Huang/ExposureNorm-Compensation)

* Extract image patches 

```python
python scripts/extract_subimages_MSEC.py
```

* Generate meta information

```python
python scripts/generate_meta_info_MSEC.py
```



### How to Test

* Download the pre-trained model

* Example: Testing on the MSEC dataset with images retouched by expert-a as ground truth

```python
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/Test_EC_MSEC_pretrained_over_expert_a_mlr.yml
```



### How to train

* Single GPU training

```python
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/Train_EC_MSEC_win8_B8G1_30k_mlr.yml
```

* Distributed training

```python
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train.py -opt options/train/Train_EC_MSEC_win8_B32G4_30k_mlr.yml --launcher pytorch
```

