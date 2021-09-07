### Pre-trained Weights (Student-Teacher)

We provide some pre-trained weights to initialize the training. The pre-trained weights are in `Student-Teacher.pth` format. Since we build our code upon [detectron2](https://github.com/facebookresearch/detectron2/), we adopt the pre-trained weights from detectron2. Here is the example to get our pre-trained weights `R50-R101.pth`.

``` 
# student
wget -c https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl

# teacher
wget -c https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl

# merge teacher and student
python merge.py
```

You can also replace the student and teacher with other pre-trained weights from [detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md). Here we provide some [pre-weights](https://drive.google.com/drive/folders/1FHaiIyIRVNBzmFM0yl4ycZGtODtXzmga?usp=sharing) that are already merged:

##### Faster R-CNN

- R18-R50
- R50-R101
- R101-R152
- MobileNetV2-R50
- EfficientNetB0-R101

##### RetinaNet

- R18-R50
- MobileNetV2-R50
- EfficientNetB0-R101