import os
import random
import torch
import numpy as np
from huggingface_hub import snapshot_download

from ...base_representation import BaseRepresentation
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
from diffusers import EulerDiscreteScheduler
from .wonder_world.core_function.key_frame_gen import KeyframeGen
from .wonder_world.marigold_lcm.marigold_pipeline import MarigoldPipeline, MarigoldNormalsPipeline
from .wonder_world.utils.utils import prepare_scheduler, load_example_yaml, convert_pt3d_cam_to_3dgs_cam, soft_stitching
from .wonder_world.utils.segment_utils import create_mask_generator_repvit
from .wonder_world.arguments import GSParams
# from ....base_models.three_dimensions.point_clouds.gaussian_splatting.scene import Scene, GaussianModel
# from ....base_models.three_dimensions.point_clouds.gaussian_splatting.gaussian_renderer import render
### wonderworld need layers predict the gaussian splatting model 
### and rendering function need change
from .wonder_world.scene import Scene, GaussianModel
from .wonder_world.gaussian_render import render
from .wonder_world.utils.loss import l1_loss, ssim
from .wonder_world.config import wonder_world_config
from ....synthesis.visual_generation.wonder_journey.wonder_world_synthesis import WonderWorldSynthesis


class WonderWorldRepresentation(BaseRepresentation):
    def __init__(self,
                 segment_model,
                 segment_processor,
                 mask_generator,
                 depth_model,
                 normal_estimator,
                 device="cuda"):
        super().__init__()
        self.segment_model = segment_model
        self.segment_processor = segment_processor
        self.mask_generator = mask_generator
        self.depth_model = depth_model
        self.normal_estimator = normal_estimator
        self.device = device

        ### need to init the keyframe generation here
        ### refer to the function self.init_keyframe_generator()
        self.keyframe_generator = None

        ### check whether the gaussian scene has been initialized
        self.gaussian_scene_init_flag = False

        ### the self.gaussians is obtained in function self.gaussian_training
        self.gaussians = None

    @classmethod
    def from_pretrained(cls,
                        segment_model_path="shi-labs/oneformer_ade20k_swin_large",
                        mask_model_path="zbhpku/repvit-sam-hf-mirror",
                        depth_predict_model_path="prs-eth/marigold-depth-v1-0",
                        normal_predict_model_path="prs-eth/marigold-normals-v1-1",
                        device="cuda",
                        **kwargs):
        """
        load segment, depth, normalization predict
        """
        ## check the model root path, utilize the snapshot download
        if os.path.isdir(segment_model_path):
            segment_model_root = segment_model_path
        else:
            print(f"Downloading weights from HuggingFace repo: {segment_model_path}")
            segment_model_root = snapshot_download(segment_model_path)
            print(f"Model downloaded to: {segment_model_root}")
        
        if os.path.isdir(mask_model_path):
            mask_model_root = mask_model_path
        else:
            print(f"Downloading weights from HuggingFace repo: {mask_model_path}")
            mask_model_root = snapshot_download(mask_model_path)
            print(f"Model downloaded to: {mask_model_root}")
        
        if os.path.isdir(depth_predict_model_path):
            depth_predict_model_root = depth_predict_model_path
        else:
            print(f"Downloading weights from HuggingFace repo: {depth_predict_model_path}")
            depth_predict_model_root = snapshot_download(depth_predict_model_path)
            print(f"Model downloaded to: {depth_predict_model_root}")
        
        if os.path.isdir(normal_predict_model_path):
            normal_predict_model_root = normal_predict_model_path
        else:
            print(f"Downloading weights from HuggingFace repo: {normal_predict_model_path}")
            normal_predict_model_root = snapshot_download(normal_predict_model_path)
            print(f"Model downloaded to: {normal_predict_model_root}")

        ## load models
        segment_processor = OneFormerProcessor.from_pretrained(segment_model_root)
        segment_model = OneFormerForUniversalSegmentation.from_pretrained(segment_model_root).to('cuda')

        mask_generator = create_mask_generator_repvit(mask_model_root)

        depth_model = MarigoldPipeline.from_pretrained(depth_predict_model_root, torch_dtype=torch.bfloat16).to(device)
        depth_model.scheduler = EulerDiscreteScheduler.from_config(depth_model.scheduler.config)
        depth_model.scheduler = prepare_scheduler(depth_model.scheduler)

        normal_estimator = MarigoldNormalsPipeline.from_pretrained(normal_predict_model_root, torch_dtype=torch.bfloat16).to(device)
        return cls(segment_model,
                   segment_processor,
                   mask_generator,
                   depth_model,
                   normal_estimator,
                   device=device)
    
    def init_keyframe_generator(self, wonder_world_synthesis: WonderWorldSynthesis):
        ## import more parameters to change the config
        config = wonder_world_config
        self.keyframe_generator = KeyframeGen(
            config=config,
            wonder_world_synthesis=wonder_world_synthesis,
            depth_model=self.depth_model,
            mask_generator=self.mask_generator,
            segment_model=self.segment_model,
            segment_processor=self.segment_processor,
            normal_estimator=self.normal_estimator,
            rotation_path=config["rotation_path"],   # for 360 camera-view generation
        )

    def train_gaussian(self,
                       gaussians,
                       scene,
                       opt: GSParams,
                       background,
                       xyz_scale,
                       initialize_scaling=True
                       ):
        iterable_gauss = range(1, opt.iterations + 1)
        trainCameras = scene.getTrainCameras().copy()
        gaussians.compute_3D_filter(cameras=trainCameras, initialize_scaling=initialize_scaling)

        for iteration in iterable_gauss:
            # 随机选择一个训练视角
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))

            # 渲染
            render_pkg = render(viewpoint_cam, gaussians, opt, background)
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

            # 计算 Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            if iteration % 100 == 0:
                print(f'Iteration {iteration}, Loss: {loss.item()}')

            loss.backward()

            # 3DGS 优化步骤 (Densification & Pruning)
            with torch.no_grad():
                n_trainable = gaussians.get_xyz.shape[0]
                viewspace_point_tensor_grad, visibility_filter, radii = viewspace_point_tensor.grad[:n_trainable], visibility_filter[:n_trainable], radii[:n_trainable]

                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                    if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        max_screen_size = opt.max_screen_size if iteration >= opt.prune_from_iter else None
                        camera_height = 0.0003 * xyz_scale
                        scene_extent = camera_height * 2 if opt.scene_extent is None else opt.scene_extent
                        opacity_lowest = 0.05
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold, opacity_lowest, scene_extent, max_screen_size)
                        gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
    
    def gaussian_training(self,
                        image,
                        sky_prompt="blue sky",
                        prompt_list=[],
                        rotation_list=[],
                        transition_list=[],
                        gen_layer=True,  # 添加layer控制参数
                        ):
        """
        执行高斯泼溅模型的训练流程
        
        Args:
            image: 初始输入图像
            prompt_list: 每个场景的文本提示列表
            rotation_list: 相机旋转路径列表
            transition_list: 场景转换参数列表
            gen_layer: 是否生成并训练layer (默认True)
        """
        xyz_scale = 1000
        background = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device=self.device)
        
        # 确保 keyframe_generator 已初始化
        if self.keyframe_generator is None:
            raise RuntimeError("Keyframe generator not initialized. Call init_keyframe_generator first.")
        
        kf_gen = self.keyframe_generator
        
        # 第一次初始化场景
        if not self.gaussian_scene_init_flag:
            print("Initializing Gaussian Scene for the first time...")
            
            # 处理初始图像
            from torchvision.transforms import ToTensor
            start_keyframe = image.convert('RGB').resize((512, 512))
            kf_gen.image_latest = ToTensor()(start_keyframe).unsqueeze(0).to(self.device)
            
            # 生成天空遮罩和点云
            sky_mask = kf_gen.generate_sky_mask().float()
            kf_gen.generate_sky_pointcloud(
                syncdiffusion_model=None,
                sky_text_prompt=sky_prompt,
                image=image,
                mask=sky_mask,
                gen_sky=False,
                style=""
            )
            
            # 设置场景名称
            scene_name = prompt_list[0] if prompt_list else "scene"
            kf_gen.recompose_image_latest_and_set_current_pc(scene_name=scene_name)
            kf_gen.increment_kf_idx()
            
            # 训练天空高斯模型
            print("Training Sky Gaussian...")
            gaussians_sky = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9)
            traindatas = kf_gen.convert_to_3dgs_traindata(xyz_scale=xyz_scale, remove_threshold=None, use_no_loss_mask=False)
            traindata_sky = traindatas[1]
            
            opt_sky = GSParams()
            opt_sky.max_screen_size = 100
            opt_sky.scene_extent = 1.5
            opt_sky.densify_from_iter = 200
            opt_sky.prune_from_iter = 200
            opt_sky.densify_grad_threshold = 1.0
            opt_sky.iterations = 399
            
            scene_sky = Scene(traindata_sky, gaussians_sky, opt_sky, is_sky=True)
            self.train_gaussian(gaussians_sky, scene_sky, opt_sky, background, xyz_scale, initialize_scaling=False)
            
            # 设置天空过滤器
            gaussians_sky.visibility_filter_all = torch.zeros(gaussians_sky.get_xyz_all.shape[0], dtype=torch.bool, device=self.device)
            gaussians_sky.delete_mask_all = torch.zeros(gaussians_sky.get_xyz_all.shape[0], dtype=torch.bool, device=self.device)
            gaussians_sky.is_sky_filter = torch.ones(gaussians_sky.get_xyz_all.shape[0], dtype=torch.bool, device=self.device)
            
            # 训练第一帧主体（包含layer处理）
            print("Training First Frame Gaussian...")
            opt = GSParams()
            
            if gen_layer:
                # 生成并训练layer
                traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(xyz_scale=xyz_scale)
                self.gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians_sky)
                scene_layer = Scene(traindata_layer, self.gaussians, opt)
                print("Training Base Layer for First Frame...")
                self.train_gaussian(self.gaussians, scene_layer, opt, background, xyz_scale)
            else:
                traindata = kf_gen.convert_to_3dgs_traindata_latest(xyz_scale=xyz_scale, use_no_loss_mask=False)
            
            # 训练主要场景
            self.gaussians = GaussianModel(sh_degree=0, previous_gaussian=self.gaussians if gen_layer else gaussians_sky)
            scene = Scene(traindata, self.gaussians, opt)
            print("Training Main Scene for First Frame...")
            self.train_gaussian(self.gaussians, scene, opt, background, xyz_scale)
            
            # 设置可见性
            tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale)
            self.gaussians.set_inscreen_points_to_visible(tdgs_cam)
            
            self.gaussian_scene_init_flag = True
            print("Gaussian Scene initialized successfully.")
        
        # 处理后续场景
        num_scenes = len(rotation_list) if rotation_list else len(prompt_list)
        
        for i in range(1, num_scenes):
            print(f"###### Processing Scene {i} / {num_scenes} ######")
            
            # 获取当前提示
            inpainting_prompt = prompt_list[i] if i < len(prompt_list) else prompt_list[-1]
            scene_name = inpainting_prompt.split(',')[0] if ',' in inpainting_prompt else inpainting_prompt
            print(f"Current Prompt: {inpainting_prompt}")
            
            # 设置关键帧参数
            kf_gen.set_kf_param(
                inpainting_resolution=kf_gen.inpainting_resolution,
                inpainting_prompt=inpainting_prompt,
                adaptive_negative_prompt=""
            )
            
            # 获取相机位姿
            current_pt3d_cam = self._get_camera_by_js_view_matrix(rotation_list[i], xyz_scale=xyz_scale,
                                                                  init_focal_length=kf_gen.init_focal_length)
            tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam, xyz_scale=xyz_scale)
            kf_gen.set_current_camera(current_pt3d_cam, archive_camera=True)
            
            # 渲染当前视角
            opt = GSParams()
            with torch.no_grad():
                render_pkg = render(tdgs_cam, self.gaussians, opt, background)
                render_pkg_nosky = render(tdgs_cam, self.gaussians, opt, background, exclude_sky=True)
            
            # 计算遮罩
            from kornia.morphology import dilation
            inpaint_mask_0p5_nosky = (render_pkg_nosky["final_opacity"] < 0.6)
            inpaint_mask_0p0_nosky = (render_pkg_nosky["final_opacity"] < 0.01)
            inpaint_mask_0p5 = (render_pkg["final_opacity"] < 0.6)
            inpaint_mask_0p0 = (render_pkg["final_opacity"] < 0.01)
            
            mask_using_full_render = torch.zeros(1, 1, 512, 512).to(self.device)
            mask_using_nosky_render = 1 - mask_using_full_render
            
            outpaint_condition_image = render_pkg_nosky["render"] * mask_using_nosky_render + render_pkg["render"] * mask_using_full_render
            fill_mask = inpaint_mask_0p5_nosky * mask_using_nosky_render + inpaint_mask_0p5 * mask_using_full_render
            outpaint_mask = inpaint_mask_0p0_nosky * mask_using_nosky_render + inpaint_mask_0p0 * mask_using_full_render
            outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7).cuda())
            
            # 执行修复
            print("Inpainting...")
            inpaint_output = kf_gen.inpaint(
                outpaint_condition_image,
                inpaint_mask=outpaint_mask,
                fill_mask=fill_mask,
                inpainting_prompt=inpainting_prompt,
                mask_strategy=np.max,
                diffusion_steps=50
            )
            
            # 处理天空和深度
            sem_seg = kf_gen.update_sky_mask()
            recomposed = soft_stitching(render_pkg["render"], kf_gen.image_latest, kf_gen.sky_mask_latest)
            
            # 深度对齐
            depth_should_be = render_pkg['median_depth'][0:1].unsqueeze(0) / xyz_scale
            mask_to_align_depth = (depth_should_be < 0.006 * 0.8) & (depth_should_be > 0.001)
            
            ground_mask = kf_gen.generate_ground_mask(sem_map=sem_seg)[None, None]
            depth_should_be_ground = kf_gen.compute_ground_depth(camera_height=0.0003)
            ground_outputable_mask = (depth_should_be_ground > 0.001) & (depth_should_be_ground < 0.006 * 0.8)
            
            joint_mask = mask_to_align_depth | (ground_mask & ground_outputable_mask)
            depth_should_be_joint = torch.where(mask_to_align_depth, depth_should_be, depth_should_be_ground)
            
            print("Estimating Depth...")
            with torch.no_grad():
                depth_guide_joint, _ = kf_gen.get_depth(
                    kf_gen.image_latest,
                    target_depth=depth_should_be_joint,
                    mask_align=joint_mask,
                    archive_output=True,
                    diffusion_steps=30,
                    guidance_steps=8
                )
            
            kf_gen.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy())
            kf_gen.image_latest = recomposed
            
            # ========== 关键的Layer处理逻辑 ========== #
            if gen_layer:
                print("Generating and processing layer...")
                # 生成layer（分离前景对象）
                kf_gen.generate_layer(pred_semantic_map=sem_seg, scene_name=scene_name)
                
                # 对layer区域重新估计深度
                depth_should_be = kf_gen.depth_latest_init
                mask_to_align_depth = ~(kf_gen.mask_disocclusion.bool()) & (depth_should_be < 0.006 * 0.8)
                mask_to_farther_depth = kf_gen.mask_disocclusion.bool() & (depth_should_be < 0.006 * 0.8)
                
                with torch.no_grad():
                    kf_gen.depth, kf_gen.disparity = kf_gen.get_depth(
                        kf_gen.image_latest,
                        archive_output=True,
                        target_depth=depth_should_be,
                        mask_align=mask_to_align_depth,
                        mask_farther=mask_to_farther_depth,
                        diffusion_steps=30,
                        guidance_steps=8
                    )
                
                # 用segment细化视差
                kf_gen.refine_disp_with_segments(
                    no_refine_mask=ground_mask.squeeze().cpu().numpy(),
                    existing_mask=~(kf_gen.mask_disocclusion).bool().squeeze().cpu().numpy(),
                    existing_disp=kf_gen.disparity_latest_init.squeeze().cpu().numpy()
                )
                
                # 修正错误深度
                wrong_depth_mask = kf_gen.depth_latest < kf_gen.depth_latest_init
                kf_gen.depth_latest[wrong_depth_mask] = kf_gen.depth_latest_init[wrong_depth_mask] + 0.0001
                kf_gen.depth_latest = kf_gen.mask_disocclusion * kf_gen.depth_latest + (1 - kf_gen.mask_disocclusion) * kf_gen.depth_latest_init
                
                # 更新天空遮罩
                kf_gen.update_sky_mask()
                
                # 更新点云 - Base部分（背景）
                valid_px_mask = outpaint_mask * (~kf_gen.sky_mask_latest)
                kf_gen.update_current_pc_by_kf(
                    image=kf_gen.image_latest,
                    depth=kf_gen.depth_latest,
                    valid_mask=valid_px_mask
                )
                
                # 更新点云 - Layer部分（前景对象）
                kf_gen.update_current_pc_by_kf(
                    image=kf_gen.image_latest_init,
                    depth=kf_gen.depth_latest_init,
                    valid_mask=kf_gen.mask_disocclusion * outpaint_mask,
                    gen_layer=True
                )
            else:
                # 不使用layer时的简单更新
                valid_px_mask = outpaint_mask * (~kf_gen.sky_mask_latest)
                kf_gen.update_current_pc_by_kf(
                    image=kf_gen.image_latest,
                    depth=kf_gen.depth_latest,
                    valid_mask=valid_px_mask
                )
            
            kf_gen.archive_latest()
            
            # ========== 训练新视角的高斯模型（包含layer） ========== #
            print(f"Training Gaussian for Scene {i}...")
            
            if gen_layer:
                # 先训练layer（前景对象）
                traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(xyz_scale=xyz_scale)
                self.gaussians = GaussianModel(sh_degree=0, previous_gaussian=self.gaussians)
                scene_layer = Scene(traindata_layer, self.gaussians, opt)
                
                print(f"Training Layer {i}...")
                self.train_gaussian(self.gaussians, scene_layer, opt, background, xyz_scale)
            else:
                traindata = kf_gen.convert_to_3dgs_traindata_latest(xyz_scale=xyz_scale, use_no_loss_mask=False)
            
            # 训练完整场景（背景+前景）
            self.gaussians = GaussianModel(sh_degree=0, previous_gaussian=self.gaussians)
            scene = Scene(traindata, self.gaussians, opt)
            
            print(f"Training Main Scene {i}...")
            self.train_gaussian(self.gaussians, scene, opt, background, xyz_scale)
            
            self.gaussians.set_inscreen_points_to_visible(tdgs_cam)
            kf_gen.increment_kf_idx()
            
            # 清理缓存
            torch.cuda.empty_cache()
        
        print("Gaussian training completed successfully.")
        return self.gaussians, kf_gen.background_image

    def gaussian_rendering(self,
                        camera_viewpoint,
                        gaussian_pc: GaussianModel=None,
                        gaussian_ply_path="",
                        ):
        """
        render the 3DGS model
        Args:
            camera_viewpoint: 相机视角矩阵 (16元素列表或4x4张量)
            gaussian_pc: 高斯模型实例 (可选)
            gaussian_ply_path: 高斯模型PLY文件路径 (可选)
        
        Returns:
            PIL Image: 渲染的RGB图像
        """
        xyz_scale = 1000
        background = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device=self.device)
        
        # 加载或使用高斯模型
        if gaussian_pc is None:
            if not gaussian_ply_path:
                if hasattr(self, 'gaussians') and self.gaussians is not None:
                    gaussians = self.gaussians
                else:
                    raise ValueError("No Gaussian model provided. Either pass gaussian_pc, gaussian_ply_path, or train a model first.")
            else:
                print(f"Loading Gaussian model from {gaussian_ply_path}")
                gaussians = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9)
                gaussians.load_ply_with_filter(gaussian_ply_path)
        else:
            gaussians = gaussian_pc
        
        # 处理相机视角
        if isinstance(camera_viewpoint, list):
            view_matrix = camera_viewpoint
        elif isinstance(camera_viewpoint, torch.Tensor):
            view_matrix = camera_viewpoint.cpu().tolist()
        else:
            raise TypeError("camera_viewpoint must be a list or torch.Tensor")
        
        # 获取相机
        current_pt3d_cam = self._get_camera_by_js_view_matrix(view_matrix, xyz_scale=xyz_scale)
        viewpoint_cam = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam, xyz_scale=xyz_scale)
        
        # 渲染
        opt = GSParams()
        with torch.no_grad():
            render_pkg = render(viewpoint_cam, gaussians, opt, background, render_visible=True)
        
        # 转换为PIL图像
        rgb = render_pkg["render"]
        pil_image = self._convert_to_rgb(rgb)
        
        return pil_image


    @torch.no_grad()
    def _get_camera_by_js_view_matrix(self, view_matrix, xyz_scale=1.0, big_view=False, init_focal_length = 960):
        """
        辅助函数: 从视图矩阵创建相机
        """
        if isinstance(view_matrix, torch.Tensor):
            view_matrix = view_matrix.reshape(4, 4).to(self.device)
        else:
            view_matrix = torch.tensor(view_matrix, device=self.device, dtype=torch.float).reshape(4, 4)
        xy_negate_matrix = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], 
                                        device=self.device, dtype=torch.float)
        view_matrix_negate_xy = view_matrix @ xy_negate_matrix
        R = view_matrix_negate_xy[:3, :3].unsqueeze(0)
        T = view_matrix_negate_xy[3, :3].unsqueeze(0)
        
        from pytorch3d.renderer import PerspectiveCameras

        K = torch.zeros((1, 4, 4), device=self.device)
        K[0, 0, 0] = init_focal_length
        K[0, 1, 1] = init_focal_length
        K[0, 0, 2] = 256
        K[0, 1, 2] = 256
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1
        
        camera = PerspectiveCameras(K=K, R=R, T=T / xyz_scale, in_ndc=False, 
                                image_size=((512, 512),), device=self.device)
        return camera


    def _convert_to_rgb(self, image_tensor):
        """
        辅助函数: 将图像张量转换为PIL图像
        """
        from PIL import Image
        image_tensor = image_tensor.clamp(0.0, 1.0)
        image_tensor = image_tensor * 255.0
        image_np = image_tensor.permute(1, 2, 0).cpu().detach().numpy()
        image_np = image_np.astype(np.uint8)
        image = Image.fromarray(image_np)
        return image

    def get_representation(self,
                           input_image,
                           sky_prompt="blue sky",
                           prompt_list=[],
                           rotation_list=[],
                           transition_list=[],
                           is_gaussian_train=True,
                           ):
        output_dict = {
            "point_cloud": None,
            "gen_background_image": None,
            "rendered_image": None,
        }
        if is_gaussian_train:
            gaussian_pc, background_image = self.gaussian_training(
                image=input_image,
                sky_prompt=sky_prompt,
                prompt_list=prompt_list,
                rotation_list=rotation_list,
                transition_list=transition_list,
                gen_layer=True,
            )
            output_dict["point_cloud"] = gaussian_pc
            output_dict["gen_background_image"] = background_image
        else:
            image = self.gaussian_rendering(
                camera_viewpoint=rotation_list[0],
            )
            output_dict["rendered_image"] = image
        return output_dict
