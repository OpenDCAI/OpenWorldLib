import os
from importlib import import_module

from .. import utils


class BasePipeline:
    def to(self, device):
        return self

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class LazyPipeline(BasePipeline):
    def __init__(self, pipeline, pipeline_info):
        self.pipeline = pipeline
        self.pipeline_info = pipeline_info
        self.device = None
        self.is_init = False

    def init_pipeline(self):
        if not self.is_init:
            self.pipeline = self.pipeline(**self.pipeline_info)
            if self.device is not None:
                self.pipeline.to(self.device)
            self.is_init = True

    def to(self, device):
        self.device = device
        if self.is_init:
            self.pipeline.to(device)
        return self

    def __call__(self, *args, **kwargs):
        self.init_pipeline()
        return self.pipeline(*args, **kwargs)


def get_vision_pipelines():
    model_dir = utils.get_model_dir()
    pipelines = {
        'depth_estimation/depth_anything/v2_small_hf': {
            '_class_name': 'vision.depth_estimation.pipeline_depth_anything.DepthAnythingPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'depth-anything/Depth-Anything-V2-Small-hf',
        },
        'depth_estimation/depth_anything/v2_base_hf': {
            '_class_name': 'vision.depth_estimation.pipeline_depth_anything.DepthAnythingPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'depth-anything/Depth-Anything-V2-Base-hf',
        },
        'depth_estimation/depth_anything/v2_large_hf': {
            '_class_name': 'vision.depth_estimation.pipeline_depth_anything.DepthAnythingPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'depth-anything/Depth-Anything-V2-Large-hf',
        },
        'depth_estimation/dpt/hybrid_midas': {
            '_class_name': 'vision.depth_estimation.pipeline_dpt.DPTForDepthEstimationPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'Intel/dpt-hybrid-midas',
        },
        'depth_estimation/dpt/large': {
            '_class_name': 'vision.depth_estimation.pipeline_dpt.DPTForDepthEstimationPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'Intel/dpt-large',
        },
        'detection/grounding_dino/swint_ogc': {
            '_class_name': 'vision.detection.pipeline_grounding_dino.GroundingDINOPipeline',
            'config_path': os.path.join(model_dir, 'grounding_dino/GroundingDINO_SwinT_OGC.py'),
            'model_path': os.path.join(model_dir, 'grounding_dino/groundingdino_swint_ogc.pth'),
        },
        'edge_detection/canny': {
            '_class_name': 'vision.edge_detection.pipeline_canny.CannyPipeline',
        },
        'edge_detection/hed/apache2': {
            '_class_name': 'vision.edge_detection.pipeline_hed.HEDPipeline',
            'model_path': 'lllyasviel/Annotators',
        },
        'edge_detection/lineart/sk_model': {
            '_class_name': 'vision.edge_detection.pipeline_lineart.LineartPipeline',
            'model_path': 'lllyasviel/Annotators',
        },
        'edge_detection/mlsd/large_512_fp32': {
            '_class_name': 'vision.edge_detection.pipeline_mlsd.MLSDPipeline',
            'model_path': 'lllyasviel/Annotators',
        },
        'edge_detection/pidinet/table5': {
            '_class_name': 'vision.edge_detection.pipeline_pidinet.PidiNetPipeline',
            'model_path': 'lllyasviel/Annotators',
        },
        'frame_interpolation/film/film_net_fp16': {
            '_class_name': 'vision.frame_interpolation.pipeline_film.FilmPipeline',
            'model_path': os.path.join(model_dir, 'frame_interpolation/film_net_fp16.pt'),
        },
        'image_restoration/prompt_ir': {
            '_class_name': 'vision.image_restoration.pipeline_prompt_ir.PromptIRPipeline',
            'model_path': os.path.join(model_dir, 'prompt_ir/model.ckpt'),
        },
        'keypoints/openpose/body_hand_face': {
            '_class_name': 'vision.keypoints.pipeline_openpose.OpenPosePipeline',
            'model_path': 'lllyasviel/Annotators',
        },
        'keypoints/rtmpose/performance': {
            '_class_name': 'vision.keypoints.pipeline_rtm_pose.RTMPosePipeline',
            'det_path': os.path.join(model_dir, 'rtmpose/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx'),
            'pose_path': os.path.join(model_dir, 'rtmpose/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx'),
            'mode': 'performance',
        },
        'optical_flow/unimatch/gmflow_scale2_regrefine6_mixdata': {
            '_class_name': 'vision.optical_flow.pipeline_unimatch.UniMatchPipeline',
            'model_path': os.path.join(model_dir, 'unimatch/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth'),
        },
        'segmentation/grounded_sam2/gd_swint_ogc_sam21_hiera_t': {
            '_class_name': 'vision.segmentation.pipeline_grounded_sam2.GroundedSAM2Pipeline',
            'gd_config_path': os.path.join(model_dir, 'grounding_dino/GroundingDINO_SwinT_OGC.py'),
            'gd_model_path': os.path.join(model_dir, 'grounding_dino/groundingdino_swint_ogc.pth'),
            'sam_config_path': 'configs/sam2.1/sam2.1_hiera_t.yaml',
            'sam_model_path': os.path.join(model_dir, 'segment_anything_2/sam2.1_hiera_tiny.pt'),
        },
        'segmentation/grounded_sam2/gd_swint_ogc_sam21_hiera_s': {
            '_class_name': 'vision.segmentation.pipeline_grounded_sam2.GroundedSAM2Pipeline',
            'gd_config_path': os.path.join(model_dir, 'grounding_dino/GroundingDINO_SwinT_OGC.py'),
            'gd_model_path': os.path.join(model_dir, 'grounding_dino/groundingdino_swint_ogc.pth'),
            'sam_config_path': 'configs/sam2.1/sam2.1_hiera_s.yaml',
            'sam_model_path': os.path.join(model_dir, 'segment_anything_2/sam2.1_hiera_small.pt'),
        },
        'segmentation/grounded_sam2/gd_swint_ogc_sam21_hiera_l': {
            '_class_name': 'vision.segmentation.pipeline_grounded_sam2.GroundedSAM2Pipeline',
            'gd_config_path': os.path.join(model_dir, 'grounding_dino/GroundingDINO_SwinT_OGC.py'),
            'gd_model_path': os.path.join(model_dir, 'grounding_dino/groundingdino_swint_ogc.pth'),
            'sam_config_path': 'configs/sam2.1/sam2.1_hiera_l.yaml',
            'sam_model_path': os.path.join(model_dir, 'segment_anything_2/sam2.1_hiera_large.pt'),
        },
        'segmentation/segment_anything/vit_b_01ec64': {
            '_class_name': 'vision.segmentation.pipeline_segment_anything.SegmentAnythingPipeline',
            'model_path': os.path.join(model_dir, 'segment_anything/sam_vit_b_01ec64.pth'),
            'model_type': 'vit_b',
        },
        'segmentation/segment_anything/vit_l_0b3195': {
            '_class_name': 'vision.segmentation.pipeline_segment_anything.SegmentAnythingPipeline',
            'model_path': os.path.join(model_dir, 'segment_anything/sam_vit_l_0b3195.pth'),
            'model_type': 'vit_l',
        },
        'segmentation/segment_anything/vit_h_4b8939': {
            '_class_name': 'vision.segmentation.pipeline_segment_anything.SegmentAnythingPipeline',
            'model_path': os.path.join(model_dir, 'segment_anything/sam_vit_h_4b8939.pth'),
            'model_type': 'vit_h',
        },
        'segmentation/upernet/convnext_tiny': {
            '_class_name': 'vision.segmentation.pipeline_upernet.UperNetForSemanticSegmentationPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'openmmlab/upernet-convnext-tiny',
        },
        'segmentation/upernet/convnext_small': {
            '_class_name': 'vision.segmentation.pipeline_upernet.UperNetForSemanticSegmentationPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'openmmlab/upernet-convnext-small',
        },
        'segmentation/upernet/convnext_large': {
            '_class_name': 'vision.segmentation.pipeline_upernet.UperNetForSemanticSegmentationPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'openmmlab/upernet-convnext-large',
        },
        'shot_boundary_detection/transnetv2': {
            '_class_name': 'vision.shot_boundary_detection.pipeline_transnetv2.TransNetV2Pipeline',
            'model_path': os.path.join(model_dir, 'transnetv2/transnetv2-pytorch-weights.pth'),
        },
    }
    return pipelines


def get_pipelines():
    pipelines_list = [get_vision_pipelines()]
    pipelines = {}
    for pipelines_i in pipelines_list:
        for key in pipelines_i:
            assert key not in pipelines
        pipelines.update(pipelines_i)
    return pipelines


def load_pipeline(pipeline_name, lazy=False, **kwargs):
    pipelines = get_pipelines()
    pipeline_info = pipelines[pipeline_name]
    pipeline_info.update(kwargs)
    parts = pipeline_info.pop('_class_name').split('.')
    module_name = '.'.join(parts[:-1])
    module = import_module('giga_models.pipelines.' + module_name)
    pipeline = getattr(module, parts[-1])
    # Download models from Hugging Face first, if necessary.
    hf_download_keys = pipeline_info.pop('_hf_download_first', [])
    for key in hf_download_keys:
        pipeline_info[key] = utils.download_from_huggingface(pipeline_info[key])
    if lazy:
        return LazyPipeline(pipeline, pipeline_info)
    else:
        return pipeline(**pipeline_info)


def list_pipelines():
    pipelines = get_pipelines()
    return list(pipelines.keys())
