# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN


def add_openset_rcnn_config(cfg):
    """
    Add config for Openset RCNN.
    """
    cfg.OPENDET_BENCHMARK = False

    cfg.MODEL.RPN.CTR_REG_LOSS_WEIGHT = 1.0
    cfg.MODEL.RPN.CTR_REG_LOSS_TYPE = "smooth_l1"
    cfg.MODEL.RPN.CTR_SMOOTH_L1_BETA = 0.0
    cfg.MODEL.RPN.IOU_THRESHOLDS_OBJECTNESS = [0.1, 0.3]
    cfg.MODEL.RPN.POSITIVE_FRACTION_OBJECTNESS = 1.0
    cfg.MODEL.RPN.NMS_THRESH_TEST = 1.0

    cfg.MODEL.ROI_BOX_HEAD.IOU_REG_LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_BOX_HEAD.IOU_REG_LOSS_TYPE = "smooth_l1"
    cfg.MODEL.ROI_BOX_HEAD.IOU_SMOOTH_L1_BETA = 0.0
    cfg.MODEL.ROI_BOX_HEAD.CLS_LOSS_WEIGHT = 1.0
    
    cfg.MODEL.ROI_HEADS.MEAN_TYPE = "geometric"
    cfg.MODEL.ROI_HEADS.OBJ_SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES = 20
    cfg.MODEL.ROI_HEADS.KNOWN_SCORE_THRESH = 0.05
    cfg.MODEL.ROI_HEADS.KNOWN_NMS_THRESH = 0.5
    cfg.MODEL.ROI_HEADS.KNOWN_TOPK = 1000
    cfg.MODEL.ROI_HEADS.UNKNOWN_SCORE_THRESH = 0.05
    cfg.MODEL.ROI_HEADS.UNKNOWN_NMS_THRESH = 0.5
    cfg.MODEL.ROI_HEADS.UNKNOWN_TOPK = 1000
    cfg.MODEL.ROI_HEADS.UNKNOWN_ID= 1000

    cfg.MODEL.PLN = CN()
    cfg.MODEL.PLN.EMD_DIM = 256
    cfg.MODEL.PLN.DISTANCE_TYPE = "COS"  # L1, L2, COS
    cfg.MODEL.PLN.REPS_PER_CLASS = 1 # 5
    cfg.MODEL.PLN.ALPHA = 0.1 # 0.3
    cfg.MODEL.PLN.BETA = 0.9 # 0.7
    cfg.MODEL.PLN.IOU_THRESHOLD = 0.5 # 0.7
    cfg.MODEL.PLN.UNK_THR = 0.4
    cfg.MODEL.PLN.LOSS_WEIGHT = 2.0
