

# DSIG

**Deep Structured Instance Graph for Distilling Object Detectors**

**Authors:** Yixin Chen, Pengguang Chen, Shu Liu, Liwei Wang, Jiaya Jia.

[[pdf]()] [[slide]()] [[supp]()] [[bibtex](#Citation)]

![](./fig/dsig.png)

This repo provides the implementation of paper "[Deep Structured Instance Graph for Distilling Object Detectors]()"(Dsig) based on [detectron2](https://github.com/facebookresearch/detectron2). Specifically, aiming at solving the feature imbalance problem while further excavating the missing relation inside semantic instances, we design a graph whose nodes correspond to instance proposal-level features and edges represent the relation between nodes. We achieve new state-of-the-art results on the COCO object detection task with diverse student-teacher pairs on both one- and two-stage detectors.

## Installation

### Requirements

- Python >= 3.6
- Pytorch >= 1.7.0
- Torchvision >= 0.8.1
- Pycocotools 2.0.2

Follow the install instructions in detectron2, note that in this repo we use detectron2 commit version `ff638c931d5999f29c22c1d46a3023e67a5ae6a1`. Download [COCO](https://cocodataset.org/) dataset and  `export DETECTRON2_DATASETS=$COCOPATH` to direct to COCO dataset. We prepare our pre-trained weights for training in `Student-Teacher` format, please follow the instructions in [Pretrained](./projects/Distillation/pretrained/README.md).

## Running 

We prepare training [configs](./projects/Distillation/configs) following the detectron2 format. For **training** a Faster R-CNN R18-FPN student with a Faster R-CNN R50-FPN teacher on 4 GPUs:

```
./start_train.sh train projects/Distillation/configs/01_Faster_RCNN_R18_R50_dsig_1x.yaml
```

For **testing**:

```
./start_train.sh eval projects/Distillation/configs/01_Faster_RCNN_R18_R50_dsig_1x.yaml
```

For **debugging**:

```
./start_train.sh debugtrain projects/Distillation/configs/01_Faster_RCNN_R18_R50_dsig_1x.yaml
```

## Results and Models

**Faster R-CNN:**

| Experiment | Schedule |  AP  | Config | Model |
| ---------- | :------: | :--: | :----: | :---: |
| R18-R50    |    1x    |      |        |       |
| R50-R101   |    1x    |      |        |       |
| R101-R152  |    1x    |      |        |       |
| MNV2-R50   |    1x    |      |        |       |
| EB0-R101   |    1x    |      |        |       |

**RetinaNet:**

| Experiment | Schedule |  AP  | Config | Model |
| ---------- | :------: | :--: | :----: | :---: |
| R18-R50    |    1x    |      |        |       |
| MNV2-R50   |    1x    |      |        |       |
| EB0-R101   |    1x    |      |        |       |

## Citation

```
​```bib
@inproceedings{chen2021dsig,
    title={Deep Structured Instance Graph for Distilling Object Detectors},
    author={Yixin Chen, Pengguang Chen, Shu Liu, Liwei Wang, and Jiaya Jia},
    booktitle={IEEE International Conference on Computer Vision (ICCV)},
    year={2021},
}
​```
```

## Contact

Please contact yxchen@cse.cuhk.edu.hk.