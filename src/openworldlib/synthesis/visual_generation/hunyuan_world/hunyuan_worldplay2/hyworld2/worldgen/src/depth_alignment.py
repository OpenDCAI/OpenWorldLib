import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from .pointcloud import point_rendering


# Custom constrained linear regression estimator
class ConstrainedLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, min_coef=1e-8, max_bias=[0.0, 1.0]):
        self.min_coef = min_coef  # Lower bound for the coefficient
        self.max_bias = max_bias
        self.model = LinearRegression()  # Base linear regression model

    def fit(self, X, y):
        # Run ordinary linear regression first
        self.model.fit(X, y)
        # Force the coefficient to be greater than min_coef
        if self.model.coef_[0] < self.min_coef:
            self.model.coef_[0] = self.min_coef  # Clamp to the lower bound if the coefficient is too small
        self.model.intercept_[0] = np.clip(self.model.intercept_[0], self.max_bias[0], self.max_bias[1])
        return self

    def predict(self, X):
        return self.model.predict(X)


def get_guided_depth_infos_v2(w2c, K, prev_points3d, prev_normal, height, width, device, render_radius=0.008):
    """
    Render depth for the current w2c and K from the guiding prev_points3d.
    Args:
        w2c: Current view w2c extrinsics [4,4]
        K: Current view intrinsics [3,3]
        prev_points3d: guiding points3d, [N,3]
        prev_normal: guiding normal, [h,w,3]
    Returns:

    """
    if prev_normal is not None:
        prev_normal_warped = prev_normal.to(device)  # Get normals for the corresponding panorama pixels
    else:
        prev_normal_warped = torch.zeros((prev_points3d.shape[0], 3), device=device)

    guided_normal, guided_depth = point_rendering(K=K[None], w2cs=w2c[None], points=prev_points3d, colors=prev_normal_warped,
                                                  h=height, w=width, render_radius=render_radius, points_per_pixel=8,
                                                  device=device, background_color=[0, 0, 0], return_depth=True)
    guided_depth = guided_depth[0, 0]
    guided_depth_mask = (guided_depth != -1)
    if prev_normal is not None:
        guided_normal = guided_normal[0]
    else:
        guided_normal = None

    return guided_depth, guided_depth_mask, guided_normal
