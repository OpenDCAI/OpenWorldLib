
import jax
import jax.numpy as jnp

from src.metrics.fid import inception


def get_detector(detector_type):
    assert detector_type in [
        "inception",
        "lpips",
    ], f"Unsupported detector type: {detector_type}. Supported types are: ['inception']"

    if detector_type == "inception":
        detector = inception.InceptionV3(pretrained=True)

        def inception_forward(
            renormalize_data=False,
        ):
            """Forward pass of the inception model to extract features."""
            params = detector.init(jax.random.PRNGKey(0), jnp.ones((1, 299, 299, 3)))

            def forward(params, x):
                if x.ndim > 4:
                    # Inception only accepts inputs in [B, H, W, C]
                    x = x.reshape(-1, *x.shape[-3:])
                if renormalize_data:
                    x = x.astype(jnp.float32) / 127.5 - 1

                x = jax.image.resize(
                    x, (x.shape[0], 299, 299, x.shape[-1]), method="bilinear"
                )
                features = detector.apply(params, x, train=False).squeeze(axis=(1, 2))

                return features

            # return params, jax.pmap(forward, axis_name='data')
            return params, jax.jit(forward)

        params, forward = inception_forward(renormalize_data=True)
        return params, forward

    else:
        raise NotImplementedError
