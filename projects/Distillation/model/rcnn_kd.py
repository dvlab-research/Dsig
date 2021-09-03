import logging
import numpy as np
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from .teacher import build_teacher
from .roi_heads import DistillROILoss
from .teacher.roi_heads import TeacherROILoss
import sys
sys.path.append('../')
from utils import mse_loss_withmask
from utils import sim_dis_compute
from utils import generate_correlation_matrix, corr_mat_mse_loss, split_features_per_image
from utils import fuse_bg_features, cat_fg_bg_features, select_topk_features_as_fg

__all__ = ["Distillation"]

@META_ARCH_REGISTRY.register()
class Distillation(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            proposal_generator: nn.Module,
            roi_heads: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            vis_period: int = 0,
            use_kd_roi_cls_loss,
            use_kd_roi_reg_loss,
            use_kd_rpn_cls_loss,
            use_kd_rpn_loc_loss,
            use_kd_feature_loss,
            use_kd_box_feature_loss,
            use_region_correlation_loss,
            use_region_correlation_bg_features,
            use_region_correlation_pool_loss,
            use_feature_roipool_fg_loss,
            use_feature_roipool_bg_loss,
            use_bg_feature_mining,
            use_bg_feature_mining_thrs,
            teacher,
            teacher_pixel_mean,
            teacher_pixel_std,
            kd_feature_loss_weight,
            kd_feature_level,
            kd_roi_cls_loss_weight,
            kd_roi_reg_loss_weight,
            kd_rpn_cls_loss_weight,
            kd_rpn_loc_loss_weight,
            region_corr_loss_weight_pool,
            feature_roipool_fg_loss_weight,
            feature_roipool_bg_loss_weight,
            smooth_l1_beta,
            bbox_reg_weights,
            box_reg_loss_type,
            rgb,
            corr_mat_sim_func,
            mine_num_bg,
            mine_bg_thrs,
            corr_mat_simloss_reduction,
            use_bg_weight_scale,
            bg_weight_alpha,
            use_semantic_pairwise_loss,
            temperature,

    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        # Teacher previous kd
        self.use_kd_roi_cls_loss = use_kd_roi_cls_loss
        self.use_kd_roi_reg_loss = use_kd_roi_reg_loss
        self.use_kd_rpn_cls_loss = use_kd_rpn_cls_loss
        self.use_kd_rpn_loc_loss = use_kd_rpn_loc_loss
        self.use_kd_feature_loss = use_kd_feature_loss
        self.use_kd_box_feature_loss = use_kd_box_feature_loss
        self.use_region_correlation_loss = use_region_correlation_loss
        self.use_region_correlation_bg_features = use_region_correlation_bg_features
        self.use_region_correlation_pool_loss = use_region_correlation_pool_loss
        self.use_feature_roipool_fg_loss = use_feature_roipool_fg_loss
        self.use_feature_roipool_bg_loss = use_feature_roipool_bg_loss

        self.use_semantic_pairwise_loss = use_semantic_pairwise_loss

        self.use_kd = self.use_kd_roi_cls_loss or \
                      self.use_kd_roi_reg_loss or \
                      self.use_kd_rpn_cls_loss or \
                      self.use_kd_rpn_loc_loss or \
                      self.use_kd_feature_loss or \
                      self.use_kd_box_feature_loss or \
                      self.use_region_correlation_loss or \
                      self.use_semantic_pairwise_loss

        if self.use_kd:
            self.teacher = teacher
            self.teacher.requires_grad = False
            self.feature_kd_loss = mse_loss_withmask

            for p in self.teacher.parameters():
                p.requires_grad = False
            self.kd_feature_loss_weight = kd_feature_loss_weight
            self.kd_feature_level = kd_feature_level
            self.kd_roi_cls_loss_weight = kd_roi_cls_loss_weight
            self.kd_roi_reg_loss_weight = kd_roi_reg_loss_weight
            self.kd_rpn_cls_loss_weight = kd_rpn_cls_loss_weight
            self.kd_rpn_loc_loss_weight = kd_rpn_loc_loss_weight

            self.region_corr_loss_weight_pool = region_corr_loss_weight_pool
            self.feature_roipool_fg_loss_weight = feature_roipool_fg_loss_weight
            self.feature_roipool_bg_loss_weight = feature_roipool_bg_loss_weight
            self.corr_mat_sim_func = corr_mat_sim_func
            self.mine_num_bg = mine_num_bg
            self.mine_bg_thrs = mine_bg_thrs
            self.corr_mat_simloss_reduction = corr_mat_simloss_reduction
            self.use_bg_feature_mining = use_bg_feature_mining
            self.use_bg_feature_mining_thrs = use_bg_feature_mining_thrs
            self.use_bg_weight_scale = use_bg_weight_scale
            self.bg_weight_alpha = bg_weight_alpha
            self.temperature = temperature

            self.smooth_l1_beta = smooth_l1_beta
            self.bbox_reg_weights = bbox_reg_weights
            self.box_reg_loss_type = box_reg_loss_type

            self.register_buffer("teacher_pixel_mean", torch.Tensor(teacher_pixel_mean).view(-1, 1, 1))
            self.register_buffer("teacher_pixel_std", torch.Tensor(teacher_pixel_std).view(-1, 1, 1))
            assert (
                    self.teacher_pixel_mean.shape == self.teacher_pixel_std.shape
            ), f"{self.teacher_pixel_mean} and {self.teacher_pixel_std} have different shapes!"

        self.rgb = rgb

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        ret = {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "use_kd_roi_cls_loss": cfg.KD.ROI_CLS_ON,
            "use_kd_roi_reg_loss": cfg.KD.ROI_REG_ON,
            "use_kd_rpn_cls_loss": cfg.KD.RPN_CLS_ON,
            "use_kd_rpn_loc_loss": cfg.KD.RPN_LOC_ON,
            "use_kd_feature_loss": cfg.KD.FEATURE_ON,
            "use_kd_box_feature_loss": cfg.KD.BOX_FEATURE_ON,
            "use_region_correlation_loss": cfg.KD.REGION_CORRELATION_LOSS_ON,
            "use_region_correlation_bg_features": cfg.KD.REGION_CORRELATION_LOSS_USE_BG_FEATURE,
            "use_region_correlation_pool_loss": cfg.KD.REGION_CORRELATION_POOL_LOSS,
            "use_feature_roipool_fg_loss": cfg.KD.FEATURE_ROIPOOL_FG_LOSS_ON,
            "use_feature_roipool_bg_loss": cfg.KD.FEATURE_ROIPOOL_BG_LOSS_ON,
            "use_bg_feature_mining": cfg.KD.BG_FEATURE_MINING_ON,
            "teacher": build_teacher(cfg),
            "kd_feature_loss_weight": cfg.KD.FEATURE_LOSS_WEIGHT,
            "kd_feature_level": cfg.KD.FEATURE_LEVEL,
            "kd_roi_cls_loss_weight": cfg.KD.ROI_CLS_LOSS_WEIGHT,
            "kd_roi_reg_loss_weight": cfg.KD.ROI_REG_LOSS_WEIGHT,
            "kd_rpn_cls_loss_weight": cfg.KD.RPN_CLS_LOSS_WEIGHT,
            "kd_rpn_loc_loss_weight": cfg.KD.RPN_LOC_LOSS_WEIGHT,
            "region_corr_loss_weight_pool": cfg.KD.REGION_CORRELATION_LOSS_WEIGHT_POOL,
            "feature_roipool_fg_loss_weight": cfg.KD.FEATURE_ROIPOOL_FG_LOSS_WEIGHT,
            "feature_roipool_bg_loss_weight": cfg.KD.FEATURE_ROIPOOL_BG_LOSS_WEIGHT,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "bbox_reg_weights": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS,
            "box_reg_loss_type": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "teacher_pixel_mean": cfg.TEACHER.MODEL.PIXEL_MEAN,
            "teacher_pixel_std": cfg.TEACHER.MODEL.PIXEL_STD,
            "rgb": cfg.MODEL.RESNETS.DEPTH < 50,
            "corr_mat_sim_func": cfg.KD.CORR_MAT_SIM_FUNCTION,
            "mine_num_bg": cfg.KD.MINE_NUM_BG,
            "corr_mat_simloss_reduction": cfg.KD.CORR_MAT_SIMLOSS_REDUCTION,
            "use_bg_feature_mining_thrs": cfg.KD.USE_BG_FEATURE_MINING_THRS,
            "mine_bg_thrs": cfg.KD.BG_MINING_THRS,
            "use_bg_weight_scale": cfg.KD.USE_BG_WEIGHT_SCALE,
            "bg_weight_alpha": cfg.KD.BG_WEIGHT_ALPHA,
            "use_semantic_pairwise_loss": cfg.KD.USE_SEMANTIC_PAIRWISE_LOSS,
            "temperature": cfg.KD.TEMPERATURE
        }

        return ret

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(
            batched_inputs, self.pixel_mean, self.pixel_std, self.rgb
        )
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # student
        features = self.backbone(images.tensor)

        # teacher
        if self.use_kd:
            teacher_images = self.teacher_preprocess_image(batched_inputs)
            with torch.no_grad():
                t_features = self.teacher.backbone(teacher_images.tensor)

        kd_losses = {}
        # kd feature loss
        if self.use_kd_feature_loss:
            feature_kd_loss = 0.
            s_features = [features[f] for f in features]
            for i, f in enumerate(t_features):
                if not i in self.kd_feature_level:
                    continue
                feature_kd_loss += self.feature_kd_loss(s_features[i], t_features[f]).mean()
            kd_losses['loss_d_f'] = feature_kd_loss * self.kd_feature_loss_weight

        # SKD
        if self.use_semantic_pairwise_loss:
            feature_semantic_pairwise_loss = 0.
            s_features = [features[f] for f in features]
            for i, f in enumerate(t_features):
                if not i in self.kd_feature_level:
                    continue
                student_f = s_features[i]
                teacher_f = t_features[f]
                student_f = torch.nn.functional.adaptive_max_pool2d(
                    student_f, [student_f.shape[2] // 4, student_f.shape[3] // 4]
                )
                teacher_f = torch.nn.functional.adaptive_max_pool2d(
                    teacher_f, [teacher_f.shape[2] // 4, teacher_f.shape[3] // 4]
                )

                feature_semantic_pairwise_loss += sim_dis_compute(student_f, teacher_f)
            kd_losses['loss_semantic_pairwise'] = feature_semantic_pairwise_loss * 0.5

        # rpn
        if self.use_kd:
            teacher_logits = self.teacher.proposal_generator(teacher_images, t_features, gt_instances)
        else:
            teacher_logits = None
        if self.proposal_generator:
            proposals, proposal_losses, kd_rpn_losses = self.proposal_generator(
                images, features, gt_instances, teacher_logits
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        if self.use_kd_rpn_cls_loss or self.use_kd_rpn_loc_loss:
            if self.use_kd_rpn_cls_loss:
                kd_losses['loss_d_rpn_cls'] = kd_rpn_losses['loss_d_rpn_cls'] * self.kd_rpn_cls_loss_weight
            if self.use_kd_rpn_loc_loss:
                kd_losses['loss_d_rpn_loc'] = kd_rpn_losses['loss_d_rpn_loc'] * self.kd_rpn_loc_loss_weight

        # roi
        detector_proposals, detector_logits, detector_losses, \
        box_features, proposals_positive, num_fg_samples, proposals_bg, num_bg_samples = self.roi_heads(
            images, features, proposals, gt_instances
        )
        if self.use_kd_roi_cls_loss or self.use_kd_roi_reg_loss:
            teacher_proposals, teacher_logits, teacher_box_features = self.teacher.roi_heads(
                teacher_images, t_features, detector_proposals
            )
            if self.use_kd_roi_cls_loss or self.use_kd_roi_reg_loss:
                kd_roi_losses = DistillROILoss(
                    self.bbox_reg_weights, detector_logits, teacher_logits, detector_proposals, self.smooth_l1_beta, self.temperature
                ).losses()
                if self.use_kd_roi_cls_loss:
                    kd_losses['loss_d_roi_cls'] = kd_roi_losses['loss_d_roi_cls'] * self.kd_roi_cls_loss_weight
                if self.use_kd_roi_reg_loss:
                    kd_losses['loss_d_roi_reg'] = kd_roi_losses['loss_d_roi_reg'] * self.kd_roi_reg_loss_weight


        # dsig
        logits_fg, box_features_fg = self.roi_heads(
            images, features, proposals_positive, None
        )

        # extract loss (cls + bbox reg)
        _, logits_fg_teacher, box_features_teacher_fg = self.teacher.roi_heads(
            teacher_images, t_features, proposals_positive
        )

        pool_features_fg = split_features_per_image(box_features_fg["pool_features"], num_fg_samples)
        pool_features_teacher_fg = split_features_per_image(box_features_teacher_fg["pool_features"],
                                                            num_fg_samples)

        # add bg features.
        if self.use_region_correlation_bg_features:
            _, box_features_bg = self.roi_heads(
                images, features, proposals_bg, None
            )
            _, bg_logits, box_features_teacher_bg = self.teacher.roi_heads(
                teacher_images, t_features, proposals_bg
            )

            pool_features_bg = split_features_per_image(box_features_bg["pool_features"], num_bg_samples)
            pool_features_teacher_bg = split_features_per_image(box_features_teacher_bg["pool_features"],
                                                                num_bg_samples)
            if self.use_bg_weight_scale:
                bg_weight_scales = [num_fg_sample / num_bg_sample * self.bg_weight_alpha
                                   for num_fg_sample, num_bg_sample in zip(num_fg_samples, num_bg_samples)]

            # bg mining
            if self.use_bg_feature_mining:
                bg_pred_class_logits = bg_logits["cls_logits"]
                bg_pred_proposal_deltas = bg_logits["proposal_deltas"]

                if self.use_bg_feature_mining_thrs:
                    bg_idx_list = TeacherROILoss(
                        self.bbox_reg_weights,
                        bg_pred_class_logits,
                        bg_pred_proposal_deltas,
                        proposals_bg,
                        self.smooth_l1_beta,
                        self.box_reg_loss_type
                    ).top_bg_idx_with_threshold(num_bg_samples, self.mine_num_bg, self.mine_bg_thrs)

                pool_features_bg_topk = select_topk_features_as_fg(pool_features_bg, bg_idx_list)
                pool_features_teacher_bg_topk = select_topk_features_as_fg(pool_features_teacher_bg, bg_idx_list)

                pool_features_fg_cat_bgmine = cat_fg_bg_features(pool_features_fg, pool_features_bg_topk)
                pool_features_teacher_fg_cat_bgmine = cat_fg_bg_features(
                    pool_features_teacher_fg, pool_features_teacher_bg_topk
                )

            # pool feat bg mse loss
            if self.use_feature_roipool_bg_loss:
                pool_features_bg_mse_loss = 0.
                if self.use_bg_weight_scale:
                    for feat_s, feat_t, bg_weight_scale in zip(pool_features_bg, pool_features_teacher_bg, bg_weight_scales):
                        pool_features_bg_mse_loss += self.feature_kd_loss(feat_s, feat_t).mean() * bg_weight_scale
                else:
                    for feat_s, feat_t in zip(pool_features_bg, pool_features_teacher_bg):
                        pool_features_bg_mse_loss += self.feature_kd_loss(feat_s, feat_t).mean()

        pool_features_fg = [torch.flatten(feat, 1) for feat in pool_features_fg]
        pool_features_teacher_fg = [torch.flatten(feat, 1) for feat in pool_features_teacher_fg]

        pool_features_fg_mse_loss = 0.
        for feat_s, feat_t in zip(pool_features_fg, pool_features_teacher_fg):
            pool_features_fg_mse_loss += self.feature_kd_loss(feat_s, feat_t).mean()

        teacher_region_correlation_matrices_pool = generate_correlation_matrix(
            pool_features_teacher_fg if not self.use_bg_feature_mining else pool_features_teacher_fg_cat_bgmine,
            simf=self.corr_mat_sim_func
        )
        student_region_correlation_matrices_pool = generate_correlation_matrix(
            pool_features_fg if not self.use_bg_feature_mining else pool_features_fg_cat_bgmine,
            simf=self.corr_mat_sim_func
        )

        region_corr_loss_pool = corr_mat_mse_loss(
            teacher_region_correlation_matrices_pool,
            student_region_correlation_matrices_pool,
            reduction=self.corr_mat_simloss_reduction
        )

        kd_losses['loss_roipool_fg'] = pool_features_fg_mse_loss * self.feature_roipool_fg_loss_weight
        if self.use_feature_roipool_bg_loss:
            kd_losses['loss_roipool_bg'] = pool_features_bg_mse_loss * self.feature_roipool_bg_loss_weight
        kd_losses['loss_region_corr_loss_pool'] = region_corr_loss_pool * self.region_corr_loss_weight_pool

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(kd_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs, self.pixel_mean, self.pixel_std, self.rgb)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return Distillation._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs, mean, std, rgb=False):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - mean) / std for x in images]
        if rgb:
            images = [x.index_select(0,torch.LongTensor([2,1,0]).to(self.device)) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def teacher_preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        return self.preprocess_image(batched_inputs, self.teacher_pixel_mean, self.teacher_pixel_std)

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"].cpu().numpy()
            assert img.shape[0] == 3, "Images should have 3 channels."
            if self.input_format == "BGR":
                img = img[::-1, :, :]
            img = img.transpose(1, 2, 0)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = " 1. GT bounding boxes  2. Predicted proposals"
            storage.put_image(vis_name, vis_img)