# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_lbvq_config(cfg):
    cfg.DATASETS.DATASET_RATIO = []

    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    # Pseudo Data Use
    cfg.INPUT.PSEUDO = CN()
    cfg.INPUT.PSEUDO.AUGMENTATIONS = ['rotation']
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768)
    cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN = 768
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice_by_clip"
    cfg.INPUT.PSEUDO.CROP = CN()
    cfg.INPUT.PSEUDO.CROP.ENABLED = False
    cfg.INPUT.PSEUDO.CROP.TYPE = "absolute_range"
    cfg.INPUT.PSEUDO.CROP.SIZE = (384, 600)

    # LSJ
    cfg.INPUT.LSJ_AUG = CN()
    cfg.INPUT.LSJ_AUG.ENABLED = False
    cfg.INPUT.LSJ_AUG.IMAGE_SIZE = 1024
    cfg.INPUT.LSJ_AUG.MIN_SCALE = 0.1
    cfg.INPUT.LSJ_AUG.MAX_SCALE = 2.0

    # LBVQ
    cfg.MODEL.LBVQ = CN()
    cfg.MODEL.LBVQ.NHEADS = 8
    cfg.MODEL.LBVQ.DROPOUT = 0.0
    cfg.MODEL.LBVQ.DIM_FEEDFORWARD = 2048
    cfg.MODEL.LBVQ.DEC_LAYERS = 6
    cfg.MODEL.LBVQ.PRE_NORM = False
    cfg.MODEL.LBVQ.HIDDEN_DIM = 256
    cfg.MODEL.LBVQ.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.LBVQ.ENFORCE_INPUT_PROJ = True

    cfg.MODEL.LBVQ.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.LBVQ.DEEP_SUPERVISION = True
    cfg.MODEL.LBVQ.LAST_LAYER_NUM = 3
    cfg.MODEL.LBVQ.MULTI_CLS_ON = True
    cfg.MODEL.LBVQ.APPLY_CLS_THRES = 0.01

    cfg.MODEL.LBVQ.SIM_USE_CLIP = True
    cfg.MODEL.LBVQ.SIM_WEIGHT = 0.5

    cfg.MODEL.LBVQ.FREEZE_DETECTOR = False
    cfg.MODEL.LBVQ.TEST_RUN_CHUNK_SIZE = 18
    cfg.MODEL.LBVQ.TEST_INTERPOLATE_CHUNK_SIZE = 5

    # SAM
    cfg.SAM = False
