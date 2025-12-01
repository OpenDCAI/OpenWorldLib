import numpy as np
from torchvision.transforms import v2
from .base_operator import BaseOperator


class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


class RecamMasterOperator(BaseOperator):
    def __init__(self,
                 operation_types=[]):
        super(RecamMasterOperator, self).__init__(operation_types=operation_types)
        camera_init = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [3390,1380,240,1]]
    
    def rotation_matrix_z(self, theta: float) -> np.ndarray:
        """绕Z轴旋转矩阵"""
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    def rotation_matrix_x(self, theta: float) -> np.ndarray:
        """绕X轴旋转矩阵"""
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]
        ])
    
    def translation_matrix(self, dx: float, dy: float, dz: float) -> np.ndarray:
        """平移矩阵"""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [dx, dy, dz, 1]
        ])

    def check_interaction(self, interaction):
        pass

    def get_interaction(self, interaction):
        """
        interaction is a dict: {"camera_viewpoint": [dx, dy, dz, theta_x, theta_x], 
                                "textual_instruction": str,}
        """
        pass

    def process_interaction(self):
        pass
