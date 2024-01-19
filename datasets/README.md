# Prepare Datasets for LBVQ

LBVQ has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  coco/
  ytvis_2019/
  ytvis_2021/
  ytvis_2022/
  ovis/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

<!-- The [model zoo](https://github.com/facebookresearch/MaskFormer/blob/master/MODEL_ZOO.md)
contains configs and models that use these builtin datasets. -->

## STEP-1: Prepare Image & Video Instance Segmentation datasets
### Expected dataset structure for [COCO](https://cocodataset.org/#download):

```
coco/
  annotations/
    instances_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

### Expected dataset structure for [YouTubeVIS 2019](https://codalab.lisn.upsaclay.fr/competitions/7682):

```
ytvis_2019/
  {train,valid,test}.json
  {train,valid,test}/
    Annotations/
    JPEGImages/
```

### Expected dataset structure for [YouTubeVIS 2021](https://codalab.lisn.upsaclay.fr/competitions/7680):

```
ytvis_2021/
  {train,valid,test}.json
  {train,valid,test}/
    JPEGImages/
```

### Expected dataset structure for [YouTubeVIS 2022](https://codalab.lisn.upsaclay.fr/competitions/3410):

```
ytvis_2022/
  {valid,test}.json
  {valid,test}/
    JPEGImages/
```

### Expected dataset structure for [OVIS](https://codalab.lisn.upsaclay.fr/competitions/4763):

```
ovis/
  annotations/
    {train,valid,test}.json
  {train,valid,test}/
```

## STEP-2: Prepare annotations for combined data
```bash
python convert_coco2ytvis.py
```
### Expected final dataset structure for all:
```
$DETECTRON2_DATASETS
+-- coco
|   |
|   +-- annotations
|   |   |
|   |   +-- instances_{train,val}2017.json
|   |   +-- coco2ytvis2019_train.json
|   |   +-- coco2ytvis2021_train.json
|   |   +-- coco2ovis_train.json
|   |
|   +-- {train,val}2017
|       |
|       +-- *.jpg
|
+-- ytvis_2019
|   ...
|
+-- ytvis_2021
|   ...
|
+-- ytvis_2022
|   ...
|
+-- ovis
    ...
```
