from enum import Enum

import jax.numpy as jnp
import numpy as np
import scipy

from src.metrics.utils import get_detector

FID = "fid"


class VideoType(Enum):
    PRED = 0
    TARGET = 1


class FIDCalculator:
    def __init__(
        self,
        num_sources,
        detector=None,
        detector_params=None,
        detector_feature_dim=2048,
    ):
        if detector is None and detector_params is None:
            # get jax.jit compiled detector
            self.detector_params, self.detector = get_detector("inception")
        elif detector is None or detector_params is None:
            raise ValueError(
                "Both `detector` and `detector_params` need to be None or both need to be NOT None."
            )

        self.total_num_images = jnp.array([[0, 0]] * num_sources, dtype=jnp.int32)

        self.num_sources = num_sources
        # Fake stats at idx = 0 | Real stats at idx = 1
        self.running_mu = jnp.zeros(
            (num_sources, 2, detector_feature_dim), dtype=jnp.float32
        )
        self.running_cov = jnp.zeros(
            (num_sources, 2, detector_feature_dim, detector_feature_dim),
            dtype=jnp.float32,
        )

    def calculate_act_stats_from_iterable(
        self,
        image_iter,
        source_idx,
        iter_type,
        mask=None,  # optional [N] bool/0-1
    ):
        """
        Extract features for a batch of images and accumulate statistics for FID.
        Features are gathered across hosts before updating statistics.
        """
        assert isinstance(image_iter, (np.ndarray, jnp.ndarray))
        assert len(image_iter.shape) == 4, "Image array should have shape (N, H, W, C)"

        batch_features = self.detector(self.detector_params, image_iter)

        iter_type_idx = iter_type.value
        # weights: 1 for valid, 0 for invalid (default all ones)
        if mask is None:
            weights = jnp.ones((batch_features.shape[0],), dtype=jnp.float32)
        else:
            weights = mask.astype(jnp.float32).reshape(-1)
            assert weights.shape[0] == batch_features.shape[0]
        n_valid = jnp.sum(weights).astype(jnp.int32)
        feats_w = batch_features * weights[:, None]  # zero out invalid rows

        self.total_num_images = self.total_num_images.at[source_idx, iter_type_idx].add(
            n_valid
        )
        self.running_mu = self.running_mu.at[source_idx, iter_type_idx].add(
            jnp.sum(feats_w, axis=0)
        )
        self.running_cov = self.running_cov.at[source_idx, iter_type_idx].add(
            feats_w.T @ feats_w
        )

    def get_act_stats(self, source_idx, iter_type):
        iter_type_idx = iter_type.value
        total_num_images = int(self.total_num_images[source_idx, iter_type_idx])

        mu = self.running_mu[source_idx, iter_type_idx] / total_num_images
        cov = (
            self.running_cov[source_idx, iter_type_idx]
            - jnp.outer(mu, mu) * total_num_images
        ) / (total_num_images - 1)
        return {"mu": mu, "sigma": cov}

    def get_fid_curve_jax(self):
        fids = []
        for source_idx in range(self.num_sources):
            fake_stats = self.get_act_stats(
                source_idx=source_idx, iter_type=VideoType.PRED
            )
            real_stats = self.get_act_stats(
                source_idx=source_idx, iter_type=VideoType.TARGET
            )

            m = jnp.square(fake_stats["mu"] - real_stats["mu"]).sum()
            s, _ = scipy.linalg.sqrtm(
                jnp.dot(fake_stats["sigma"], real_stats["sigma"]), disp=False
            )
            fids.append(
                float(
                    jnp.real(
                        m + jnp.trace(fake_stats["sigma"] + real_stats["sigma"] - s * 2)
                    )
                )
            )
        fid = sum(fids) / self.num_sources
        return fid


def calculate_metrics_from_batch(
    preds,
    targets,
    metrics_to_cal,
    real_lengths,
    fid_calculator=None,
):
    """
    Calculate metrics for a single batch of predictions and targets.
    All calculators, extractors, and curve functions must be pre-initialized and passed in.
    Returns only the batch metrics dictionary.
    """
    B, T, P, H, W, C = preds.shape

    if FID in metrics_to_cal:
        assert fid_calculator is not None
        valid_mask = (jnp.arange(T)[None, :] < real_lengths[:, None]).reshape(-1)

        for player_idx in range(P):
            pred_imgs = preds[..., player_idx, :, :, :].reshape(B * T, H, W, C)
            target_imgs = targets[..., player_idx, :, :, :].reshape(B * T, H, W, C)

            fid_calculator.calculate_act_stats_from_iterable(
                image_iter=pred_imgs,
                source_idx=player_idx,
                iter_type=VideoType.PRED,
                mask=valid_mask,
            )
            fid_calculator.calculate_act_stats_from_iterable(
                image_iter=target_imgs,
                source_idx=player_idx,
                iter_type=VideoType.TARGET,
                mask=valid_mask,
            )
