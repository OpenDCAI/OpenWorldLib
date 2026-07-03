import dm_pix
import jax.numpy as jnp
from flax import nnx


def ssim(
    preds_BTHWC,  # [0, 255] image in uint8
    target_BTHWC,  # [0, 255] image in uint8
    kernel_size=11,
    sigma=1.5,
    k1=0.01,
    k2=0.03,
):
    assert (
        preds_BTHWC.shape == target_BTHWC.shape and preds_BTHWC.ndim == 5
    ), "preds and target must have the same shape and be 5D [B, T, H, W, C] tensors!"
    B, T, H, W, C = preds_BTHWC.shape
    assert C == 3
    normalized_preds_BTHWC, normalized_target_BTHWC = (
        jnp.float32(preds_BTHWC) / 255.0,
        jnp.float32(target_BTHWC) / 255.0,
    )

    # scan is pretty good. you can specify what your batch size is for memory purposes.
    @nnx.scan(in_axes=(1, 1), out_axes=0, unroll=10)
    def _ssim(normalized_preds_BHWC, normalized_target_BHWC):
        ssim_vals = dm_pix.ssim(
            normalized_preds_BHWC,
            normalized_target_BHWC,
            max_val=1.0,
            filter_size=kernel_size,
            filter_sigma=sigma,
            k1=k1,
            k2=k2,
            return_map=False,  # Return scalar values, not SSIM maps
        )
        ssim = jnp.nanmean(ssim_vals)  # average over batch
        return ssim

    return _ssim(normalized_preds_BTHWC, normalized_target_BTHWC)


def psnr(
    preds_BTHWC,
    targets_BTHWC,
    base=10.0,
    data_range=255.0,
    eps=1e-10,
):
    B, T, _, _, _ = preds_BTHWC.shape
    assert (
        preds_BTHWC.shape == targets_BTHWC.shape and preds_BTHWC.ndim == 5
    ), "preds and targets must have the same shape and be 5D [B, T, H, W, C] tensors!"

    preds_fp32_BTCHW = preds_BTHWC.astype(jnp.float32)
    targets_fp32_BTCHW = targets_BTHWC.astype(jnp.float32)

    mse_BT = (
        jnp.mean(jnp.square(preds_fp32_BTCHW - targets_fp32_BTCHW), axis=(2, 3, 4))
        + eps
    )
    psnr_base_e = 2 * jnp.log(data_range) - jnp.log(mse_BT)
    psnr_vals_BT = psnr_base_e * (10 / jnp.log(base))
    assert psnr_vals_BT.shape == (
        B,
        T,
    ), "PSNR values must have the same shape as the input tensors!"
    psnr_per_timestep_T = jnp.nanmean(psnr_vals_BT, axis=0)
    return psnr_per_timestep_T
