###
# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
# All rights reserved.
###
import random

from ..arguments import GSParams
from .dataset_readers import readDataInfo
from .gaussian_model import WonderWorldGaussianModel as GaussianModel


class Scene:
    def __init__(self, traindata, gaussians: GaussianModel, opt: GSParams, is_sky: bool = False):
        self.traindata = traindata
        self.gaussians = gaussians
        
        info = readDataInfo(traindata, opt.white_background)
        # random.shuffle(info.train_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        self.train_cameras = info.train_cameras

        self.gaussians.create_from_pcd(info.point_cloud, self.cameras_extent, is_sky=is_sky)
        self.gaussians.training_setup(opt)

    def getTrainCameras(self):
        return self.train_cameras