import random
from collections import defaultdict
from typing import Any, Optional, List, Dict

import os
import gc
import hydra
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig
from torchmetrics import MeanMetric
from torch.distributed import all_gather_object
from safetensors.torch import load_file as load_safetensors

from mirror_training.utils.logger import RankedLogger
from mirror_training.utils.scheduler import MultiLinearWarmupCosineAnnealingLR
from mirror_training.utils.misc import (
    compose_batches_from_list,
    convert_defaultdict_to_dict,
    deep_merge_dict,
)
from mirror_training.utils.eval.pointmap_eval import umeyama, accuracy, completion
from mirror_training.utils.eval.camera_pose_eval import (
    se3_to_relative_pose_error,
    calculate_auc,
)
from mirror_training.utils.eval.normal_eval import get_normal_error, get_normal_metrics
from mirror_training.utils.eval.nvs_eval import RenderingMetrics
from mirror_training.utils.eval.depthmap_eval import get_depth_metrics, EVAL_DEPTH_METADATA
from mirror_training.utils.viz import save_novel_view_render, log_training_input_and_output_images

from mirror_src.models.models.worldmirror import WorldMirror

from mirror_src.models.utils.camera_utils import vector_to_extrinsics

log = RankedLogger(__name__, rank_zero_only=True)


class WorldMirrorWrapper(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        pretrained: Optional[str] = None,
        resume: Optional[str] = None,
        optimizer: Optional[DictConfig] = None,
        scheduler: Optional[DictConfig] = None,
        enable_cond: bool = True,
        cond_sampling_strategy: str = "uniform",
        cond_sampling_probs: Optional[List[float]] = None,
        train_criterion: Optional[DictConfig] = None,
        eval_modalities: List[str] = [],
        vis_log_dir: str = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler
        self.train_criterion = train_criterion
        self.enable_cond = enable_cond
        self.cond_sampling_strategy = cond_sampling_strategy
        self.cond_sampling_probs = cond_sampling_probs
        self.pretrained = pretrained
        self.resume = resume
        self.eval_modalities = eval_modalities
        self.vis_log_dir = vis_log_dir

        self.point_map_results = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )
        self.camera_pose_results = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )
        self.normal_results = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )
        self.nvs_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.depthmap_results = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )
        self.tau_list = [5, 15, 30]
        self.render_metrics = RenderingMetrics()

    def setup(self, stage: str):
        # resume training from checkpoint
        if self.resume is not None:
            log.info(f"resume training from checkpoint: {self.resume}")
            return

        # load pretrained weights
        log.info(f"\n{'-' * 60}\n{'Loading pretrained weights'.center(60)}\n{'-' * 60}")
        if self.pretrained is not None:
            log.info(f"Loading pretrained weights from {self.pretrained}")
            if self.pretrained.endswith(".safetensors"):
                ckpt_state_dict = load_safetensors(self.pretrained)
            elif self.pretrained.endswith(".ckpt"):
                ckpt_state_dict = torch.load(self.pretrained, map_location=torch.device('cpu'))
                if 'state_dict' in ckpt_state_dict:
                    ckpt_state_dict = ckpt_state_dict['state_dict']
                    ckpt_state_dict = {k.replace("model.", ""): v for k, v in ckpt_state_dict.items()} 
            else:
                raise ValueError(f"Unsupported checkpoint format: {self.pretrained}")

            current_state_dict = self.model.state_dict()

            matched_keys = []
            mismatched_keys = []
            not_found_keys = []
            for key in current_state_dict.keys():
                if key in ckpt_state_dict:
                    if current_state_dict[key].shape == ckpt_state_dict[key].shape:
                        current_state_dict[key] = ckpt_state_dict[key]
                        matched_keys.append(key)
                    else:
                        log.warning(
                            f"Shape mismatch for key '{key}': "
                            f"current {current_state_dict[key].shape} vs "
                            f"pretrained {ckpt_state_dict[key].shape}"
                        )
                        mismatched_keys.append(key)
                else:
                    not_found_keys.append(key)

            self.model.load_state_dict(current_state_dict, strict=False)

            log.info(f"✓ Loaded {len(matched_keys)}/{len(current_state_dict)} keys")
            num_params = sum(p.numel() for p in self.model.parameters())
            log.info(
                f"✓ Loaded {num_params / 1e6:.2f}M parameters from {self.pretrained}"
            )
            if not_found_keys:
                log.info(
                    f"⚠ {len(not_found_keys)} parameters not found: {not_found_keys[:5]}..."
                )

            # clean up
            del ckpt_state_dict, current_state_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # assert not self.training, "Load pretrained weights with WorldMirror.from_pretrained() only allowed in evaluation mode"
            log.info("Loading weights from Hugging Face")
            if isinstance(self.model, WorldMirror):
                config = self.model.config
                self.model = WorldMirror.from_pretrained("/opt/tiger/WMflow/ckpts", local_files_only=True)
                for key in config.keys():
                    setattr(self.model, key, config[key])
            else:
                raise ValueError(f"Unsupported model type: {type(self.model)}")
            num_params = sum(p.numel() for p in self.model.parameters())
            log.info(f"✓ Loaded {num_params / 1e6:.2f}M parameters from Hugging Face")

    def configure_optimizers(self):
        freeze_params = self.optimizer_config.get("freeze_params", [])
        log.info(f"Freeze parameter: {freeze_params}")
        for name, param in self.model.named_parameters():
            for freeze_param in freeze_params:
                if freeze_param in name:
                    param.requires_grad = False

        if "param_groups" in self.optimizer_config:
            # build optimizer param groups
            param_groups = self._build_param_groups(
                self.optimizer_config.param_groups,
                self.optimizer_config.get("new_params", []),
                self.optimizer_config.get("pretrained_params", []),
            )
            if self.optimizer_config.type == "AdamW":
                optimizer_class = torch.optim.AdamW
            else:
                raise ValueError(
                    f"Unsupported optimizer type: {self.optimizer_config.type}"
                )
            optimizer_args = {}
            if hasattr(self.optimizer_config, "betas"):
                optimizer_args["betas"] = tuple(
                    float(beta) for beta in self.optimizer_config.betas
                )
            if hasattr(self.optimizer_config, "eps"):
                optimizer_args["eps"] = float(self.optimizer_config.eps)

            optimizer = optimizer_class(param_groups, **optimizer_args)
            log.info(f"Initialized Optimizer: {optimizer_class.__name__}")
            # build scheduler
            total_steps = self.trainer.estimated_stepping_batches
            scheduler_config = self.scheduler_config
            warmup_steps = scheduler_config.get("warmup_steps", 0)
            eta_min_ratio = scheduler_config.get("eta_min_ratio", 0.1)
            max_steps = scheduler_config.get("max_steps", 0)
            if self.scheduler_config.type == "WarmupCosineAnnealingLR":
                scheduler_class = MultiLinearWarmupCosineAnnealingLR
            else:
                raise ValueError(
                    f"Unsupported scheduler type: {self.scheduler_config.type}"
                )
            scheduler = scheduler_class(
                optimizer=optimizer,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                eta_min_ratio=eta_min_ratio,
            )
            log.info(f"Initialized Scheduler: {scheduler_class.__name__}")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "train/lr",
                },
            }
        else:
            return {
                "optimizer": hydra.utils.instantiate(
                    self.optimizer_config, params=self.parameters()
                )
            }

    def _build_param_groups(
        self, param_groups_config, new_params_list, pretrained_params_list
    ):
        """Build parameter groups for optimizer.

        Args:
            param_groups_config: Parameter group configuration
            new_params_list: List of new parameter module names
            pretrained_params_list: List of pretrained parameter module names

        Returns:
            list: List of parameter groups
        """
        pretrained_params = []
        backbone_params = []
        new_params = []
        self.param_mapping = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_new = any(new_name in name for new_name in new_params_list)
            is_pretrained = any(
                pretrained_name in name for pretrained_name in pretrained_params_list
            )
            if is_new:
                new_params.append(param)
            elif is_pretrained:
                pretrained_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []
        if pretrained_params and "pretrained" in param_groups_config:
            param_groups.append(
                {"params": pretrained_params, **param_groups_config.pretrained}
            )
            self.param_mapping["pretrained"] = len(param_groups) - 1
        if backbone_params and "backbone" in param_groups_config:
            param_groups.append(
                {"params": backbone_params, **param_groups_config.backbone}
            )
            self.param_mapping["backbone"] = len(param_groups) - 1
        if new_params and "new" in param_groups_config:
            param_groups.append({"params": new_params, **param_groups_config.new})
            self.param_mapping["new"] = len(param_groups) - 1
        assert len(param_groups) > 0

        return param_groups

    def forward(self, x: Dict[str, torch.Tensor], cond_flags: List[int]):
        return self.model(x, cond_flags=cond_flags, is_inference=False)

    def on_train_start(self):
        log.info(f"\n{'-' * 60}\n{'Training Start'.center(60)}\n{'-' * 60}")
        log.info(f"Number of DDP processes: {torch.distributed.get_world_size()}")
        log.info(f"Number of GPUs: {torch.cuda.device_count()}")
        log.info(f"Number of CPUs: {os.cpu_count()}")

    def on_train_epoch_start(self):
        if hasattr(self.trainer.train_dataloader, "dataset") and hasattr(
            self.trainer.train_dataloader.dataset, "set_epoch"
        ):
            self.trainer.train_dataloader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.train_dataloader, "batch_sampler") and hasattr(
            self.trainer.train_dataloader.batch_sampler, "set_epoch"
        ):
            self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)

    def training_step(self, batch: List[Dict[str, torch.Tensor]], batch_idx: int):
        # compose batches
        batched_inputs = compose_batches_from_list(batch, device=self.device)

        # forward pass with condition flags
        if self.enable_cond:
            if self.cond_sampling_strategy == "uniform":
                cond_flags = [random.randint(0, 1) for _ in range(3)]
            elif self.cond_sampling_strategy == "custom":
                assert (
                    self.cond_sampling_probs is not None
                ), "Custom sampling probabilities are not set"
                cond_flags = [
                    random.random() < prob for prob in self.cond_sampling_probs
                ]
            else:
                raise ValueError(
                    f"Unsupported sampling strategy: {self.cond_sampling_strategy}"
                )
        else:
            cond_flags = [0, 0, 0]

        preds = self.forward(batched_inputs, cond_flags=cond_flags)

        # loss calculation
        assert self.train_criterion is not None, "Train criterion is not set"
        with torch.amp.autocast(dtype=torch.float32, device_type=self.device.type):
            loss, loss_dict = self.train_criterion(batched_inputs, preds)

        # check for nan or inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            log.error(f"NaN or Inf detected in loss at batch {batch_idx}")
            log.error(f"Loss value: {loss.item()}")
            log.error(f"Loss dict: {loss_dict}")
            raise ValueError(
                f"NaN or Inf detected in loss at batch {batch_idx}. Please check the data and model."
            )

        # log training metrics
        epoch_fraction = (
            self.current_epoch + (batch_idx + 1) / self.trainer.num_training_batches
        )
        self.log("train/loss_all", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(
            "trainer/epoch",
            torch.tensor(epoch_fraction),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        # log optimizer lr
        for param_name, param_index in self.param_mapping.items():
            self.log(
                f"trainer/lr_{param_name}",
                self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[
                    param_index
                ],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        # log detailed loss components
        for loss_name, loss_value in loss_dict.items():
            if isinstance(loss_value, torch.Tensor):
                self.log(
                    f"train/{loss_name}",
                    loss_value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )
        
        # log training inputs and outputs
        if self.trainer.is_global_zero:
            log_training_input_and_output_images(preds, batched_inputs, self.logger, self.global_step, save_path=self.vis_log_dir,)
        return loss

    def on_train_epoch_end(self):
        pass

    def on_validation_start(self):
        log.info(f"\n{'-' * 60}\n{'Evaluation Start'.center(60)}\n{'-' * 60}")
        log.info(f"Number of DDP processes: {torch.distributed.get_world_size()}")
        log.info(f"Number of GPUs: {torch.cuda.device_count()}")
        log.info(f"Number of CPUs: {os.cpu_count()}")

    def on_validation_epoch_start(self):
        for loader in self.trainer.val_dataloaders:
            if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(0)
            if hasattr(loader, "batch_sampler") and hasattr(
                loader.batch_sampler, "set_epoch"
            ):
                loader.batch_sampler.set_epoch(0)
            if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(0)

    def validation_step(
        self,
        batch: List[Dict[str, torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        # compose batches
        batched_inputs = compose_batches_from_list(
            batch, device=self.device, validation=True
        )

        # forward pass with optional conditions
        preds_all = {}
        preds_all["nocond"] = self.forward(batched_inputs, cond_flags=[0, 0, 0])
        if self.enable_cond:
            if "camera_poses" in batched_inputs:
                preds_all["wcam"] = self.forward(batched_inputs, cond_flags=[1, 0, 0])
            if "depthmap" in batched_inputs:
                preds_all["wdepth"] = self.forward(batched_inputs, cond_flags=[0, 1, 0])
            if "camera_intrs" in batched_inputs:
                preds_all["wintrs"] = self.forward(batched_inputs, cond_flags=[0, 0, 1])
            if "camera_poses" in batched_inputs and "camera_intrs" in batched_inputs:
                if "depthmap" in batched_inputs:
                    preds_all["wall"] = self.forward(
                        batched_inputs, cond_flags=[1, 1, 1]
                    )
                else:
                    preds_all["wcam_wpose"] = self.forward(
                        batched_inputs, cond_flags=[1, 0, 1]
                    )

        dataset = batched_inputs["dataset"][0]
        # eval point_map reconstruction
        if "point_map" in self.eval_modalities and dataset in [
            "dtu",
            "7scenes",
            "nrgbd",
        ]:
            self.eval_point_map(preds_all, batched_inputs)

        if "camera_pose" in self.eval_modalities and dataset in ["Re10K_Pose"]:
            self.eval_camera_pose(preds_all, batched_inputs)

        if "normal" in self.eval_modalities and dataset in [
            "nyuv2",
            "scannet_normal",
            "ibims",
        ]:
            self.eval_normal(preds_all, batched_inputs)

        if "nvs" in self.eval_modalities and dataset in [
            "re10k_nvs_test",
            "dl3dv_nvs_test",
            "hypersim",
        ]:
            self.eval_nvs(preds_all, batched_inputs)
            if self.trainer.is_global_zero and batch_idx < 4:
                save_novel_view_render(
                    self.model.gs_renderer,
                    preds_all,
                    batched_inputs,
                    self.logger,
                    self.global_step,
                    save_path=self.vis_log_dir,
                )

        if "depth_map" in self.eval_modalities and dataset in [
            "nyuv2_monodepth",
            "sintel_monodepth",
            "sintel_videodepth",
            "kitti_videodepth",
        ]:
            self.eval_depthmap(preds_all, batched_inputs)

        del batched_inputs, preds_all
        torch.cuda.empty_cache()
        return 0

    def eval_point_map(self, preds_all, batched_inputs):
        for cond_name, preds in preds_all.items():
            scene_name = "/".join(batched_inputs["label"][0].split("/")[:-1])

            colors = (
                batched_inputs["img"][0].permute(0, 2, 3, 1).cpu().numpy()
            )  # [S, H, W, 3]
            gt_pts = batched_inputs["pts3d"][0].cpu().numpy()  # [S, H, W, 3]
            valid_mask = batched_inputs["valid_mask"][0].cpu().numpy()  # [S, H, W]
            pred_pts = preds["pts3d"][0].cpu().numpy()  # [S, H, W, 3]

            assert (
                pred_pts.shape == gt_pts.shape
            ), "Predicted and ground truth point maps have different shapes"

            # coarse align
            c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
            pred_pts = c * np.einsum("nhwj, ij -> nhwi", pred_pts, R) + t.T

            # filter invalid points
            pred_pts = pred_pts[valid_mask].reshape(-1, 3)
            gt_pts = gt_pts[valid_mask].reshape(-1, 3)
            colors = colors[valid_mask].reshape(-1, 3)

            # get pred and gt point clouds
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred_pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)

            # ICP alignment
            dataset = batched_inputs["dataset"][0]
            threshold = 100 if "dtu" in dataset else 0.1

            trans_init = np.eye(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd,
                pcd_gt,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
            transformation = reg_p2p.transformation
            pcd = pcd.transform(transformation)

            # compute metrics
            pcd.estimate_normals()
            pcd_gt.estimate_normals()
            pred_normals = np.asarray(pcd.normals)
            gt_normals = np.asarray(pcd_gt.normals)
            pred_pts_np = np.asarray(pcd.points)
            gt_pts_np = np.asarray(pcd_gt.points)

            # compute metrics
            acc, acc_med, nc1, nc1_med = accuracy(
                gt_pts_np, pred_pts_np, gt_normals, pred_normals, device=self.device
            )
            comp, comp_med, nc2, nc2_med = completion(
                gt_pts_np, pred_pts_np, gt_normals, pred_normals, device=self.device
            )
            self.point_map_results[cond_name][dataset][scene_name] = {
                "accuracy": acc,
                "accuracy_median": acc_med,
                "completion": comp,
                "completion_median": comp_med,
                "nc1": nc1,
                "nc1_median": nc1_med,
                "nc2": nc2,
                "nc2_median": nc2_med,
            }

    def eval_camera_pose(self, preds_all, batched_inputs):
        for cond_name, preds in preds_all.items():
            scene_name = batched_inputs["label"][0]
            gt_camera_pose = batched_inputs["camera_poses"][0]  # [S, 4, 4] | w2c
            pred_camera_params = preds["camera_params"][0]  # [S, 7]
            pred_camera_pose = vector_to_extrinsics(
                pred_camera_params
            )  # [S, 3, 4] | w2c
            pred_camera_pose = torch.cat(
                [
                    pred_camera_pose,
                    torch.tensor([0, 0, 0, 1], device=self.device)
                    .view(1, 1, 4)
                    .repeat(pred_camera_pose.shape[0], 1, 1),
                ],
                dim=1,
            )  # [S, 4, 4] | w2c
            dataset = batched_inputs["dataset"][0]
            assert (
                gt_camera_pose.shape == pred_camera_pose.shape
            ), "Ground truth and predicted camera poses have different shapes"

            # compute metrics
            rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
                pred_camera_pose, gt_camera_pose, len(pred_camera_pose)
            )

            for tau in self.tau_list:
                self.camera_pose_results[cond_name][dataset][scene_name][
                    f"RRA_{tau}"
                ] = ((rel_rangle_deg < tau).float().mean().item())
                self.camera_pose_results[cond_name][dataset][scene_name][
                    f"RTA_{tau}"
                ] = ((rel_tangle_deg < tau).float().mean().item())

            self.camera_pose_results[cond_name][dataset][scene_name]["AUC_30"] = (
                calculate_auc(rel_rangle_deg, rel_tangle_deg).item()
            )

    def eval_normal(self, preds_all, batched_inputs):
        for cond_name, preds in preds_all.items():
            scene_name = "/".join(batched_inputs["label"][0].split(".")[:-1])

            gt_normal = batched_inputs["normals"][0]  # [S, H, W, 3]
            gt_normal_norm = torch.linalg.norm(gt_normal, dim=-1, keepdim=True)
            gt_normal_mask = (gt_normal_norm > 0.5) & (gt_normal_norm < 1.5)
            pred_normal = -1 * preds["normals"][0]  # [S, H, W, 3]
            dataset = batched_inputs["dataset"][0]

            normal_error = get_normal_error(gt_normal, pred_normal)
            total_normal_errors = normal_error[gt_normal_mask]
            metrics = get_normal_metrics(total_normal_errors)

            self.normal_results[cond_name][dataset][scene_name] = {
                "mean": metrics["mean"],
                "median": metrics["median"],
                "rmse": metrics["rmse"],
                "a1": metrics["a1"],
                "a2": metrics["a2"],
                "a3": metrics["a3"],
                "a4": metrics["a4"],
                "a5": metrics["a5"],
            }

    def eval_nvs(self, preds_all, batched_inputs):
        self.render_metrics.lpips_fn.to(self.device)
        assert "is_target" in batched_inputs
        context_views = (batched_inputs["is_target"][0] == False).sum().item()
        for cond_name, preds in preds_all.items():
            scene_name = "/".join(batched_inputs["label"][0].split(".")[-3:])
            gt_colors = preds["gt_colors"][0, context_views:]  # [S, H, W, 3]
            pred_colors = preds["rendered_colors"][0, context_views:]  # [S, H, W, 3]
            masks = preds["valid_masks"][0, context_views:]  # Shape: [S, H, W]
            dataset = batched_inputs["dataset"][0]

            # compute metrics
            if preds["gt_colors"][0].shape[0] > context_views:
                nvs_results = self.render_metrics(pred_colors, gt_colors, masks)
            else:
                nvs_results = {"psnr": 0, "ssim": 0, "lpips": 0}

            self.nvs_results[cond_name][dataset][scene_name]["PSNR"] = nvs_results[
                "psnr"
            ]
            self.nvs_results[cond_name][dataset][scene_name]["SSIM"] = nvs_results[
                "ssim"
            ]
            self.nvs_results[cond_name][dataset][scene_name]["LPIPS"] = nvs_results[
                "lpips"
            ]

    def eval_depthmap(self, preds_all, batched_inputs):
        for cond_name, preds in preds_all.items():
            scene_name = batched_inputs["label"][0]
            dataset = batched_inputs["dataset"][0]
            gathered_depth_metrics = []
            for idx in range(batched_inputs["depthmap"].shape[1]):
                gt_depth = batched_inputs["depthmap"][0, idx].cpu().numpy()  # [H, W]
                pred_depth = preds["depth"][0, idx].detach().cpu().numpy()  # [H, W]
                pred_depth = cv2.resize(
                    pred_depth,
                    (gt_depth.shape[1], gt_depth.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
                depth_results, error_map, depth_predict, depth_gt = get_depth_metrics(
                    pred_depth,
                    gt_depth,
                    **EVAL_DEPTH_METADATA[dataset]["depth_evaluation_kwargs"],
                )
                gathered_depth_metrics.append(depth_results)

            average_metrics = {
                key: np.average(
                    [metrics[key] for metrics in gathered_depth_metrics],
                    weights=[
                        metrics["valid_pixels"] for metrics in gathered_depth_metrics
                    ],
                ).item()
                for key in gathered_depth_metrics[0].keys()
                if key != "valid_pixels"
            }

            self.depthmap_results[cond_name][dataset][scene_name] = {
                "Abs Rel": average_metrics["Abs Rel"],
                "δ < 1.25": average_metrics["δ < 1.25"],
                "RMSE": average_metrics["RMSE"],
            }

    def on_validation_epoch_end(self):
        self.merge_log_pointmap_results()
        self.merge_log_camera_pose_results()
        self.merge_log_normal_results()
        self.merge_log_nvs_results()
        self.merge_log_depthmap_results()

    def merge_log_pointmap_results(self):
        # merge results from all GPUs to the main process
        if torch.distributed.is_initialized():
            plain_dict_results = convert_defaultdict_to_dict(self.point_map_results)
            all_gpu_results = [None] * torch.distributed.get_world_size()
            all_gather_object(all_gpu_results, plain_dict_results)

            if self.trainer.is_global_zero:
                merged_results = {}
                for gpu_results in all_gpu_results:
                    deep_merge_dict(merged_results, gpu_results)
                self.point_map_results = merged_results

        # only log the results on the main process
        if not torch.distributed.is_initialized() or self.trainer.is_global_zero:
            for name, datasets in self.point_map_results.items():
                for dataset, scenes in datasets.items():
                    # Extract metrics from all scenes
                    metrics_lists = {
                        "accuracy": [],
                        "accuracy_median": [],
                        "completion": [],
                        "completion_median": [],
                        "nc1": [],
                        "nc1_median": [],
                        "nc2": [],
                        "nc2_median": [],
                    }

                    for scene_metrics in scenes.values():
                        for metric_name in metrics_lists.keys():
                            metrics_lists[metric_name].append(
                                scene_metrics[metric_name]
                            )

                    # Compute and log mean values for each metric
                    for metric_name, values in metrics_lists.items():
                        mean_value = np.mean(values)
                        self.log(
                            f"val_recon_{name}_{dataset}/{metric_name}",
                            float(mean_value),
                        )

        # Clear all dataset metrics after logging
        self.point_map_results.clear()
        self.point_map_results = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )

    def merge_log_camera_pose_results(self):
        # merge results from all GPUs to the main process
        if torch.distributed.is_initialized():
            plain_dict_results = convert_defaultdict_to_dict(self.camera_pose_results)
            all_gpu_results = [None] * torch.distributed.get_world_size()
            all_gather_object(all_gpu_results, plain_dict_results)

            if self.trainer.is_global_zero:
                merged_results = {}
                for gpu_results in all_gpu_results:
                    deep_merge_dict(merged_results, gpu_results)
                self.camera_pose_results = merged_results

        # only log the results on the main process
        if not torch.distributed.is_initialized() or self.trainer.is_global_zero:
            for name, datasets in self.camera_pose_results.items():
                for dataset, scenes in datasets.items():
                    # Extract metrics from all scenes
                    metrics_lists = {
                        "RRA_5": [],
                        "RRA_15": [],
                        "RRA_30": [],
                        "RTA_5": [],
                        "RTA_15": [],
                        "RTA_30": [],
                        "AUC_30": [],
                    }

                    for scene_name, metrics in scenes.items():
                        for metric_name in metrics_lists.keys():
                            metrics_lists[metric_name].append(metrics[metric_name])

                    # Compute and log mean values for each metric
                    for metric_name, values in metrics_lists.items():
                        mean_value = np.mean(values)
                        self.log(f"val_pose_{name}_{dataset}/{metric_name}", mean_value)

        # Clear all dataset metrics after logging
        self.camera_pose_results.clear()
        self.camera_pose_results = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )

    def merge_log_normal_results(self):
        # merge results from all GPUs to the main process
        if torch.distributed.is_initialized():
            plain_dict_results = convert_defaultdict_to_dict(self.normal_results)
            all_gpu_results = [None] * torch.distributed.get_world_size()
            all_gather_object(all_gpu_results, plain_dict_results)

            if self.trainer.is_global_zero:
                merged_results = {}
                for gpu_results in all_gpu_results:
                    deep_merge_dict(merged_results, gpu_results)
                self.normal_results = merged_results

        # only log the results on the main process
        if not torch.distributed.is_initialized() or self.trainer.is_global_zero:
            for name, datasets in self.normal_results.items():
                for dataset, scenes in datasets.items():
                    # Extract metrics from all scenes
                    metrics_lists = {
                        "mean": [],
                        "median": [],
                        "rmse": [],
                        "a1": [],
                        "a2": [],
                        "a3": [],
                        "a4": [],
                        "a5": [],
                    }

                    for scene_name, metrics in scenes.items():
                        for metric_name in metrics_lists.keys():
                            metrics_lists[metric_name].append(metrics[metric_name])

                    # Compute and log mean values for each metric
                    for metric_name, values in metrics_lists.items():
                        mean_value = np.mean(values)
                        self.log(
                            f"val_normal_{name}_{dataset}/{metric_name}", mean_value
                        )

        # Clear all dataset metrics after logging
        self.normal_results.clear()
        self.normal_results = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )

    def merge_log_nvs_results(self):
        # merge results from all GPUs to the main process
        if torch.distributed.is_initialized():
            plain_dict_results = convert_defaultdict_to_dict(self.nvs_results)
            all_gpu_results = [None] * torch.distributed.get_world_size()
            all_gather_object(all_gpu_results, plain_dict_results)

            if self.trainer.is_global_zero:
                merged_results = {}
                for gpu_results in all_gpu_results:
                    deep_merge_dict(merged_results, gpu_results)
                self.nvs_results = merged_results

        # only log the results on the main process
        if not torch.distributed.is_initialized() or self.trainer.is_global_zero:
            for name, datasets in self.nvs_results.items():
                for dataset, scenes in datasets.items():
                    # Extract metrics from all scenes
                    metrics_lists = {
                        "PSNR": [],
                        "SSIM": [],
                        "LPIPS": [],
                    }

                    for scene_name, metrics in scenes.items():
                        for metric_name in metrics_lists.keys():
                            metrics_lists[metric_name].append(metrics[metric_name])

                    # Compute and log mean values for each metric
                    for metric_name, values in metrics_lists.items():
                        mean_value = np.mean(values)
                        self.log(f"val_nvs_{name}_{dataset}/{metric_name}", mean_value)

        # Clear all dataset metrics after logging
        self.nvs_results.clear()
        self.nvs_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    def merge_log_depthmap_results(self):
        # merge results from all GPUs to the main process
        if torch.distributed.is_initialized():
            plain_dict_results = convert_defaultdict_to_dict(self.depthmap_results)
            all_gpu_results = [None] * torch.distributed.get_world_size()
            all_gather_object(all_gpu_results, plain_dict_results)

            if self.trainer.is_global_zero:
                merged_results = {}
                for gpu_results in all_gpu_results:
                    deep_merge_dict(merged_results, gpu_results)
                self.depthmap_results = merged_results

        # only log the results on the main process
        if not torch.distributed.is_initialized() or self.trainer.is_global_zero:
            for name, datasets in self.depthmap_results.items():
                for dataset, scenes in datasets.items():
                    # Extract metrics from all scenes
                    metrics_lists = {
                        "Abs Rel": [],
                        "δ < 1.25": [],
                        "RMSE": [],
                    }

                    for scene_name, metrics in scenes.items():
                        for metric_name in metrics_lists.keys():
                            metrics_lists[metric_name].append(metrics[metric_name])

                    # Compute and log mean values for each metric
                    for metric_name, values in metrics_lists.items():
                        mean_value = np.mean(values)
                        self.log(
                            f"val_depthmap_{name}_{dataset}/{metric_name}", mean_value
                        )

        # Clear all dataset metrics after logging
        self.depthmap_results.clear()
        self.depthmap_results = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )
