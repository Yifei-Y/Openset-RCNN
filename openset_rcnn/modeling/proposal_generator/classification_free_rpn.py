# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple, Union
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling import build_anchor_generator, build_rpn_head, RPN_HEAD_REGISTRY, PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.box_regression import Box2BoxTransformLinear
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels

from ..box_regression_w_iou import _dense_box_regression_loss_w_iou
from ..find_top_proposals import find_top_rpn_proposals


"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    B: size of the box parameterization

Naming convention:

    deltas: refers to the 4-d (l, t, r, b) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransformLinear`).

    centerness: the localization quality target described in :paper:`FCOS`.

    gt_labels: ground-truth binary classification labels for objectness

    pred_anchor_deltas: predicted box2box transform deltas

    gt_anchor_deltas: ground-truth box2box transform deltas

    pred_centerness: predicted centerness

    gt_centerness: ground-truth centerness
"""


@RPN_HEAD_REGISTRY.register()
class ClsFreeRPNHead(nn.Module):
    """
    Classification-Free RPN bounding box regression and localization quality prediction heads.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts 
    bounding-box deltas specifying how to deform each anchor into an object proposal 
    and a second 1x1 conv predicts centerness specifying localization quality.
    """

    @configurable
    def __init__(
        self, *, in_channels: int, num_anchors: int, box_dim: int = 4, conv_dims: List[int] = (-1,)
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_channels (int): number of input feature channels. When using multiple
                input features, they must have the same number of channels.
            num_anchors (int): number of anchors to predict for *each spatial position*
                on the feature map. The total number of anchors for each
                feature map will be `num_anchors * H * W`.
            box_dim (int): dimension of a box, which is also the number of box regression
                predictions to make for each anchor. An axis aligned box has
                box_dim=4, while a rotated box has box_dim=5.
            conv_dims (list[int]): a list of integers representing the output channels
                of N conv layers. Set it to -1 to use the same number of output channels
                as input channels.
        """
        super().__init__()
        cur_channels = in_channels
        # Keeping the old variable names and structure for backwards compatiblity.
        # Otherwise the old checkpoints will fail to load.
        if len(conv_dims) == 1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            # 3x3 conv for the hidden representation
            self.conv = self._get_rpn_conv(cur_channels, out_channels)
            cur_channels = out_channels
        else:
            self.conv = nn.Sequential()
            for k, conv_dim in enumerate(conv_dims):
                out_channels = cur_channels if conv_dim == -1 else conv_dim
                if out_channels <= 0:
                    raise ValueError(
                        f"Conv output channels should be greater than 0. Got {out_channels}"
                    )
                conv = self._get_rpn_conv(cur_channels, out_channels)
                self.conv.add_module(f"conv{k}", conv)
                cur_channels = out_channels
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=1, stride=1)
        # 1x1 conv for predicting centerness
        self.centerness = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)

        # Keeping the order of weights initialization same for backwards compatiblility.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def _get_rpn_conv(self, in_channels, out_channels):
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        return {
            "in_channels": in_channels,
            "num_anchors": num_anchors[0],
            "box_dim": box_dim,
            "conv_dims": cfg.MODEL.RPN.CONV_DIMS,
        }

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps

        Returns:
            list[Tensor]: A list of L elements. 
                Element i is a tensor of shape (N, A*box_dim, Hi, Wi) representing 
                the predicted "deltas" used to transform anchors to proposals.
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted centerness for all anchors.
        """
        pred_anchor_deltas = []
        pred_centerness = []
        for x in features:
            t = self.conv(x)
            t = F.normalize(t, p=2, dim=1)
            pred_anchor_deltas.append(self.anchor_deltas(t))
            pred_centerness.append(self.centerness(t).sigmoid())
        return pred_anchor_deltas, pred_centerness


@PROPOSAL_GENERATOR_REGISTRY.register()
class ClsFreeRPN(nn.Module):
    """
    Classification-Free Region Proposal Network.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Matcher,
        objectness_anchor_matcher: Matcher,
        box2box_transform: Box2BoxTransformLinear,
        batch_size_per_image: int,
        positive_fraction: float,
        objectness_positive_fraction: float,
        pre_nms_topk: Tuple[float, float],
        post_nms_topk: Tuple[float, float],
        nms_thresh: Tuple[float, float],
        min_box_size: float = 0.0,
        anchor_boundary_thresh: float = -1.0,
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        box_reg_loss_type: str = "smooth_l1",
        box_reg_smooth_l1_beta: float = 0.0,
        ctr_reg_loss_type: str = "smooth_l1",
        ctr_reg_smooth_l1_beta: float = 0.0
    ):
        """
        Args:
            in_features (list[str]): list of names of input features to use
            head (nn.Module): a module that predicts regression deltas and centerness
                for each level from a list of per-level features
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            anchor_matcher (Matcher): label the anchors by matching them with ground truth for 
                box regression.
            objectness_anchor_matcher (Matcher): label the anchors by matching them with ground truth 
                for objectness prediction.
            box2box_transform (Box2BoxTransformLinear): defines the distances from the anchor center 
                to the four sides of the instance box.
            batch_size_per_image (int): number of anchors per image to sample for training.
            positive_fraction (float): fraction of foreground anchors to sample for training box regression.
            objectness_positive_fraction (float): fraction of foreground anchors to sample for 
                training objectness prediction.
            pre_nms_topk (tuple[float]): (train, test) that represents the
                number of top k proposals to select before NMS, in
                training and testing.
            post_nms_topk (tuple[float]): (train, test) that represents the
                number of top k proposals to select after NMS, in
                training and testing.
            nms_thresh (tuple[float]): NMS threshold used to de-duplicate the predicted proposals, 
                in training and testing.
            min_box_size (float): remove proposal boxes with any side smaller than this threshold,
                in the unit of input image pixels
            anchor_boundary_thresh (float): legacy option
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all rpn losses together, or a dict of individual weightings. Valid dict keys are:
                    "loss_rpn_loc" - applied to box regression loss
                    "loss_rpn_ctr" - applied to centerness regression loss
            box_reg_loss_type (str): Loss type to use for box regression. 
                Supported losses: "smooth_l1", "iou", "giou", "diou", "ciou".
            box_reg_smooth_l1_beta (float): beta parameter for the smooth L1 box regression loss. 
                Default to use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
            ctr_reg_loss_type (str): Loss type to use for centerness regression. 
                Supported losses: "smooth_l1".
            ctr_reg_smooth_l1_beta (float): beta parameter for the smooth L1 centerness regression loss. 
                Default to use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
        """
        super().__init__()
        self.in_features = in_features
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.matcher = Matcher(
            [0.5], [0,1], allow_low_quality_matches=False
        )
        self.objectness_anchor_matcher = objectness_anchor_matcher
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.objectness_positive_fraction = objectness_positive_fraction
        # Map from self.training state to train/test settings
        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self.nms_thresh = {True: nms_thresh[0], False: nms_thresh[1]}
        self.min_box_size = float(min_box_size)
        self.anchor_boundary_thresh = anchor_boundary_thresh
        if isinstance(loss_weight, float):
            loss_weight = {"loss_rpn_loc": loss_weight, 
                           "loss_rpn_ctr": loss_weight}
        self.loss_weight = loss_weight
        self.box_reg_loss_type = box_reg_loss_type
        self.box_reg_smooth_l1_beta = box_reg_smooth_l1_beta
        self.ctr_reg_loss_type = ctr_reg_loss_type
        self.ctr_reg_smooth_l1_beta = ctr_reg_smooth_l1_beta

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.RPN.IN_FEATURES
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "objectness_positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION_OBJECTNESS,
            "loss_weight": {
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_ctr": cfg.MODEL.RPN.CTR_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT
            },
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            "box2box_transform": Box2BoxTransformLinear(normalize_by_size=True),
            "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
            "box_reg_smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,
            "ctr_reg_loss_type": cfg.MODEL.RPN.CTR_REG_LOSS_TYPE,
            "ctr_reg_smooth_l1_beta": cfg.MODEL.RPN.CTR_SMOOTH_L1_BETA,
        }

        ret["nms_thresh"] = (cfg.MODEL.RPN.NMS_THRESH, cfg.MODEL.RPN.NMS_THRESH_TEST)
        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)

        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["objectness_anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS_OBJECTNESS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features])
        return ret

    def _subsample_labels(self, label, pos_frac):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
            pos_frac (float): fraction of positives.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, pos_frac, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels for bbox regression 
                whose length is the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class. (When sampling, the unchosen 0 and 1 will be 
                changed to -1)
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels for centerness regress 
                whose length is the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class. (When sampling, the unchosen 0 and 1 will be 
                changed to -1)
            list[Tensor]:
                List of #img tensors. i-th element is a vector whose length is R. 
                The values are the centerness for each anchor. 
                Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors) # Boxes: sum(Hi * Wi * num_cell_anchor_i), i: feature level

        gt_boxes = [x.gt_boxes for x in gt_instances] # list[Boxes]: num_images, num_gt_boxes_i
        image_sizes = [x.image_size for x in gt_instances] # list[tuple]: num_images, 2
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        objectness_gt_labels = []
        gt_centerness = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes): # images
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors) # Tensor: (num_gt_boxes, num_anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            objectness_matched_idxs, objectness_gt_labels_i = retry_if_cuda_oom(self.objectness_anchor_matcher)(match_quality_matrix)
            # Tensor: num_anchors, [0, num_gt_boxes)
            # Tensor: num_anchors, {-1, 0, 1}
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            objectness_gt_labels_i = objectness_gt_labels_i.to(device=gt_boxes_i.device)
            # del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1
                objectness_gt_labels_i[~anchors_inside_image] = -1

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i, self.positive_fraction)  # Tensor: num_anchors, {-1, 0, 1}
            objectness_gt_labels_i = self._subsample_labels(objectness_gt_labels_i, self.objectness_positive_fraction)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            matched_pairwise_dist = self.box2box_transform.get_deltas(anchors.tensor, gt_boxes_i.tensor[objectness_matched_idxs]) # Tensor: (num_anchors, 4), ltrb
            matched_pairwise_dist = matched_pairwise_dist[:, [0,2,1,3]] # Tensor: (num_anchors, 4), lrtb
            is_in_boxes = (matched_pairwise_dist >= 0).all(dim=1)
            matched_pairwise_dist[~is_in_boxes, :] = 0
            left_right = matched_pairwise_dist[:,0:2]
            top_bottom = matched_pairwise_dist[:,2:4]
            gt_centerness_i = torch.sqrt(
                (torch.min(left_right, -1)[0] / (torch.max(left_right, -1)[0] + 1e-12)) * 
                (torch.min(top_bottom, -1)[0] / (torch.max(top_bottom, -1)[0] + 1e-12)))
            gt_centerness_i[objectness_gt_labels_i == 0] = 0.0
            
            del match_quality_matrix

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
            objectness_gt_labels.append(objectness_gt_labels_i)
            gt_centerness.append(gt_centerness_i)

        return gt_labels, matched_gt_boxes, objectness_gt_labels, gt_centerness

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        pred_centerness: List[torch.Tensor],
        gt_centerness: List[torch.Tensor],
        objectness_gt_labels: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_centerness (list[Tensor]): A list of L elements. 
                Element i is a tensor of shape (N, Hi*Wi*A) representing the predicted "centerness" 
                used to measure the localization quality.
            gt_centerness (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            objectness_gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_loc` for proposal localization and 
                `loss_rpn_ctr` for centerness prediction.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        gt_centerness = torch.stack(gt_centerness) # (N, sum(Hi*Wi*Ai))
        objectness_gt_labels = torch.stack(objectness_gt_labels)

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        obj_pos_mask = objectness_gt_labels == 1
        obj_mask = objectness_gt_labels != -1
        obj_num_pos_anchors = obj_pos_mask.sum().item()
        obj_num_neg_anchors = (objectness_gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)
        storage.put_scalar("rpn/obj_num_pos_anchors", obj_num_pos_anchors / num_images)
        storage.put_scalar("rpn/obj_num_neg_anchors", obj_num_neg_anchors / num_images)

        localization_loss = _dense_box_regression_loss_w_iou(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.box_reg_smooth_l1_beta,
        )

        if self.ctr_reg_loss_type == "smooth_l1":
            centerness_loss = smooth_l1_loss(
                cat(pred_centerness, dim=1)[obj_mask],
                gt_centerness[obj_mask],
                beta=self.ctr_reg_smooth_l1_beta,
                reduction="sum"
            )

        normalizer = self.batch_size_per_image * num_images
        losses = {
            # The original Faster R-CNN paper uses a slightly different normalizer
            # for loc loss. But it doesn't matter in practice
            "loss_rpn_loc": localization_loss / normalizer,
            "loss_rpn_ctr": centerness_loss / normalizer
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features] # list[Tensor]: levels, (images, channels_i, Hi, Wi)
        anchors = self.anchor_generator(features) # list[Boxes]: levels, Hi x Wi x num_cell_anchors_i

        pred_anchor_deltas, pred_centerness = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ] # list[Tensor]: levels, (images, Hi * Wi * num_cell_anchors_i, 4); 
        pred_centerness = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            centerness.permute(0, 2, 3, 1).flatten(1)
            for centerness in pred_centerness
        ] # list[Tensor]: levels, (images, Hi * Wi * num_cell_anchors_i)

        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes, objectness_gt_labels, gt_centerness = self.label_and_sample_anchors(anchors, gt_instances)
            # fore_samples: iou > h_threshold and center is in box and random sample 
            # back_samples: iou < l_threshold and random sample
            # ignore: l_threshold < iou < h_threshold or center is not in box or unsampled fore/back
            losses = self.losses(
                anchors, gt_labels, 
                pred_anchor_deltas, gt_boxes,
                pred_centerness, gt_centerness,
                objectness_gt_labels
            )
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_anchor_deltas, pred_centerness, images.image_sizes
        )
        
        num_proposals = 0
        for proposals_i in proposals:
            num_proposals += len(proposals_i)
        if self.training:
            storage = get_event_storage()
            storage.put_scalar("rpn/num_proposals", num_proposals / len(proposals))

        return proposals, losses

    def predict_proposals(
        self,
        anchors: List[Boxes],
        pred_anchor_deltas: List[torch.Tensor],
        pred_centerness: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses.
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            
            return find_top_rpn_proposals(
                pred_proposals,
                pred_centerness,
                image_sizes,
                self.nms_thresh[self.training],
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training,
            )

    def _decode_proposals(self, anchors: List[Boxes], pred_anchor_deltas: List[torch.Tensor]):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
        """
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals
