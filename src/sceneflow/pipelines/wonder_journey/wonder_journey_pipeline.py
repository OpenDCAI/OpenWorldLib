import gc
import copy
from datetime import datetime
from pathlib import Path
import torch
from torchvision.transforms import ToPILImage
from omegaconf import OmegaConf


sys.path.append("../../operators")
from wonder_journey_oprator import WonderJourneyOperator
sys.path.append("../../representations/models/wonder_journey")
from representation_wonder_journey import VisualRepresentation
sys.path.append("../../synthesis/visual_generation/wonder_journey")
from synthesis_wonder_journey import VisualSynthesis
sys.path.append("../../reasoning")
from wonder_journey_reasoning import PromptReasoning
sys.path.append("../../representations/models/wonder_journey/wonder_journey")
from util.finetune_utils import finetune_depth_model, finetune_decoder
from util.general_utils import save_video
from util.utils import save_depth_map, merge_frames, merge_keyframes

class WonderJourneyPipeline:
    def __init__(self, representation_model, reasoning_model, synthesis_model, config):
        self.representation_model = representation_model
        self.reasoning_model = reasoning_model
        self.synthesis_model = synthesis_model
        self.config = config

    @classmethod
    def from_pretrained(cls, config, oneformer_path, sd_path, depth_model_path="dpt_beit_large_512.pt"):
        """
        加载所有子模块
        """
        device = config["device"]

        print("Loading Representation Models...")
        representation_model = VisualRepresentation.from_pretrained(
            oneformer_path=oneformer_path,
            depth_model_path=depth_model_path,
            device=device
        )

        print("Loading Synthesis Models...")
        synthesis_model = VisualSynthesis.from_pretrained(
            pretrained_model_path=sd_path,
            device=device
        )

        print("Loading Reasoning Models...")
        reasoning_model = PromptReasoning.from_pretrained(
            runs_dir=config['runs_dir'],
            is_list_control_text=isinstance(config.get('control_text', []), list),
            device=device
        )

        return cls(representation_model, reasoning_model, synthesis_model, config)

    @staticmethod
    def create_operator(config):
        """
        辅助函数：帮助 test.py 创建 Operator，而不需要 test.py 导入 Operator 类
        """
        operator = WonderJourneyOperator(config)
        operator.seeding(config["seed"])
        return operator

    def evaluate(self, model):
        fps = model.config["save_fps"]
        save_root = Path(model.run_dir)

        video = (255 * torch.cat(model.images, dim=0)).to(torch.uint8).detach().cpu()
        video_reverse = (255 * torch.cat(model.images[::-1], dim=0)).to(torch.uint8).detach().cpu()

        save_video(video, save_root / "output.mp4", fps=fps)
        save_video(video_reverse, save_root / "output_reverse.mp4", fps=fps)

    def evaluate_epoch(self, model, epoch, vmax=None):
        rendered_depth = model.rendered_depths[epoch].clamp(0).cpu().numpy()
        depth = model.depths[epoch].clamp(0).cpu().numpy()
        save_root = Path(model.run_dir) / "images"
        save_root.mkdir(exist_ok=True, parents=True)
        (save_root / "inpaint_input_image").mkdir(exist_ok=True, parents=True)
        (save_root / "frames").mkdir(exist_ok=True, parents=True)
        (save_root / "masks").mkdir(exist_ok=True, parents=True)
        (save_root / "post_masks").mkdir(exist_ok=True, parents=True)
        (save_root / "rendered_images").mkdir(exist_ok=True, parents=True)
        (save_root / "rendered_depths").mkdir(exist_ok=True, parents=True)
        (save_root / "depth").mkdir(exist_ok=True, parents=True)

        model.inpaint_input_image[epoch].save(save_root / "inpaint_input_image" / f"{epoch}.png")
        ToPILImage()(model.images[epoch][0]).save(save_root / "frames" / f"{epoch}.png")
        ToPILImage()(model.masks[epoch][0]).save(save_root / "masks" / f"{epoch}.png")
        ToPILImage()(model.post_masks[epoch][0]).save(save_root / "post_masks" / f"{epoch}.png")
        ToPILImage()(model.rendered_images[epoch][0]).save(save_root / "rendered_images" / f"{epoch}.png")
        save_depth_map(rendered_depth, save_root / "rendered_depths" / f"{epoch}.png", vmax=vmax)
        save_depth_map(depth, save_root / "depth" / f"{epoch}.png", vmax=vmax, save_clean=True)

        if hasattr(model, "outter_masks"):
            (save_root / "outter_masks").mkdir(exist_ok=True, parents=True)
            ToPILImage()(model.outter_masks[epoch]).save(save_root / "outter_masks" / f"{epoch}.png")
        if epoch == 0:
            with open(Path(model.run_dir) / "config.yaml", "w") as f:
                OmegaConf.save(model.config, f)

    def empty_cache(self):
        torch.cuda.empty_cache()
        gc.collect()

    def process(self, operator):
        config = self.config
        
        if config['skip_gen']:
            kfgen_save_folder = Path(config['runs_dir']) / f"{config['kfgen_load_dt_string']}_kfgen"
        else:
            dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            kfgen_save_folder = Path(config['runs_dir']) / f"{dt_string}_kfgen"
        kfgen_save_folder.mkdir(exist_ok=True, parents=True)
        
        cutoff_depth = config['fg_depth_range'] + config['depth_shift']
        vmax = cutoff_depth * 2
        inpainting_resolution_gen = config['inpainting_resolution_gen']

        # Get initial data from operator
        interaction_data = operator.process_interaction()
        start_keyframe = operator.process_perception()
        
        content_prompt = interaction_data["content_prompt"]
        style_prompt = interaction_data["style_prompt"]
        adaptive_negative_prompt = interaction_data["adaptive_negative_prompt"]
        background_prompt = interaction_data["background_prompt"]
        control_text = interaction_data["control_text"]
        scene_name = interaction_data["scene_name"]
        entities = interaction_data["entities"]
        rotation_path = interaction_data["rotation_path"]
        inpainting_prompt = interaction_data["inpainting_prompt"]
        
        scene_dict = {'scene_name': scene_name, 'entities': entities, 'style': style_prompt, 'background': background_prompt}
        
        all_keyframes = [start_keyframe]
        all_rundir = []

        assert len(rotation_path) >= config['num_scenes'] * config['num_keyframes']

        # Extract models
        pt_gen = self.reasoning_model
        inpainter_pipeline = self.synthesis_model.inpainter_pipeline
        vae = self.synthesis_model.vae
        mask_generator = self.synthesis_model.mask_generator
        segment_model = self.representation_model.oneformer_model
        segment_processor = self.representation_model.oneformer_processor
        depth_model = self.representation_model.depth_model
        
        # We need to import KeyframeGen/Interp inside here or at top level. 
        # Since they are in representation.py, we imported them at the top.
        from representation import KeyframeGen, KeyframeInterp

        ###### ------------------ Main loop ------------------ ######

        for i in range(config['num_scenes']):
            # GPT logic commented out as per your request
            # if config['use_gpt']:
            #     control_text_this = control_text[i] if isinstance(control_text, list) else None
            #     scene_dict = pt_gen.run_conversation(...)
            # inpainting_prompt = pt_gen.generate_prompt(...)
            
            for j in range(config['num_keyframes']):

                ###### ------------------ Keyframe generation ------------------ ######
                if config['skip_gen']:
                    kf_gen_dict = torch.load(kfgen_save_folder / f"s{i:02d}_k{j:01d}_gen_dict.pt")
                    kf1_depth, kf2_depth = kf_gen_dict['kf1_depth'], kf_gen_dict['kf2_depth']
                    kf1_image, kf2_image = kf_gen_dict['kf1_image'], kf_gen_dict['kf2_image']
                    kf1_camera, kf2_camera = kf_gen_dict['kf1_camera'], kf_gen_dict['kf2_camera']
                    kf2_mask = kf_gen_dict['kf2_mask']
                    inpainting_prompt, adaptive_negative_prompt = kf_gen_dict['inpainting_prompt'], kf_gen_dict['adaptive_negative_prompt']
                    rotation = kf_gen_dict['rotation']
                else:
                    rotation = rotation_path[i*config['num_keyframes'] + j]
                    regen_negative_prompt = ""
                    config['inpainting_resolution_gen'] = inpainting_resolution_gen
                    for regen_id in range(config['regenerate_times'] + 1):
                        if regen_id > 0:
                            operator.seeding(-1)
                        
                        kf_gen = KeyframeGen(config, inpainter_pipeline, mask_generator, depth_model, vae, rotation, 
                                            start_keyframe, inpainting_prompt, regen_negative_prompt + adaptive_negative_prompt,
                                            segment_model=segment_model, segment_processor=segment_processor).to(config["device"])
                        save_root = Path(kf_gen.run_dir) / "images"
                        kf_idx = 0

                        save_depth_map(kf_gen.depths[kf_idx].detach().cpu().numpy(), save_root / 'kf1_original', vmin=0, vmax=vmax)
                        kf_gen.refine_disp_with_segments(kf_idx, background_depth_cutoff=cutoff_depth)
                        save_depth_map(kf_gen.depths[kf_idx].detach().cpu().numpy(), save_root / 'kf1_processed', vmin=0, vmax=vmax)
                        self.evaluate_epoch(kf_gen, kf_idx, vmax=vmax)

                        kf_idx = 1
                        render_output = kf_gen.render(kf_idx)
                        inpaint_output = kf_gen.inpaint(render_output["rendered_image"], render_output["inpaint_mask"])

                        regenerate = False # Force False for now

                        if not regenerate:
                            break

                        kf_gen.depth_model = None 
                        self.empty_cache()

                    if config["finetune_decoder_gen"]:
                        ToPILImage()(inpaint_output["inpainted_image"].detach()[0]).save(save_root / 'kf2_before_ft.png')
                        finetune_decoder(config, kf_gen, render_output, inpaint_output, config['num_finetune_decoder_steps'])

                    kf_gen.update_images_and_masks(inpaint_output["latent"], render_output["inpaint_mask"])

                    kf2_depth_should_be = render_output['rendered_depth']
                    mask_to_align_depth = ~(render_output["inpaint_mask_512"]>0) & (kf2_depth_should_be < cutoff_depth + kf_gen.kf_delta_t)
                    mask_to_cutoff_depth = ~(render_output["inpaint_mask_512"]>0) & (kf2_depth_should_be >= cutoff_depth + kf_gen.kf_delta_t)

                    if config["finetune_depth_model"]:
                        finetune_depth_model(config, kf_gen, kf2_depth_should_be, kf_idx, mask_align=mask_to_align_depth, 
                                            mask_cutoff=mask_to_cutoff_depth, cutoff_depth=cutoff_depth + kf_gen.kf_delta_t)
                    with torch.no_grad():
                        kf2_ft_depth_original, kf2_ft_disp_original = kf_gen.get_depth(kf_gen.images[kf_idx])
                        kf_gen.depths.append(kf2_ft_depth_original), kf_gen.disparities.append(kf2_ft_disp_original)
                    
                    kf_gen.depth_model = None
                    self.empty_cache()

                    kf_gen.refine_disp_with_segments(kf_idx, background_depth_cutoff=cutoff_depth + kf_gen.kf_delta_t)
                    save_depth_map(kf_gen.depths[-1].cpu().numpy(), save_root / 'kf2_ft_depth_processed', vmin=0, vmax=vmax)
                        
                    kf_gen.vae.decoder = copy.deepcopy(kf_gen.decoder_copy)
                    self.evaluate_epoch(kf_gen, kf_idx, vmax=vmax)

                    start_keyframe = ToPILImage()(kf_gen.images[1][0])
                    all_keyframes.append(start_keyframe)

                    kf1_depth, kf2_depth = kf_gen.depths[0], kf_gen.depths[-1]
                    kf1_image, kf2_image = kf_gen.images[0], kf_gen.images[1]
                    kf1_camera, kf2_camera = kf_gen.cameras[0], kf_gen.cameras[1]
                    kf2_mask = render_output["inpaint_mask_512"]
                    kf_gen_dict = {'kf1_depth': kf1_depth, 'kf2_depth': kf2_depth, 'kf1_image': kf1_image, 'kf2_image': kf2_image, 
                                'kf1_camera': kf1_camera, 'kf2_camera': kf2_camera, 'kf2_mask': kf2_mask, 'inpainting_prompt': inpainting_prompt, 
                                'adaptive_negative_prompt': adaptive_negative_prompt, 'rotation': rotation}
                    torch.save(kf_gen_dict, kfgen_save_folder / f"s{i:02d}_k{j:01d}_gen_dict.pt")

                    if config['skip_interp']:
                        kf_gen = kf_gen.to('cpu')
                        del kf_gen
                        self.empty_cache()
                        continue

                ###### ------------------ Keyframe interpolation ------------------ ######
                
                is_last_scene = i == config['num_scenes'] - 1
                is_last_keyframe = j == config['num_keyframes'] - 1
                try:
                    is_next_rotation = rotation_path[i*config['num_keyframes'] + j + 1] != 0
                except IndexError:
                    is_next_rotation = False
                try:
                    is_previous_rotation = rotation_path[i*config['num_keyframes'] + j - 1] != 0
                except IndexError:
                    is_previous_rotation = False
                is_beginning = i == 0 and j == 0
                speed_up = (rotation == 0) and ((is_last_scene and is_last_keyframe) or is_next_rotation)
                speed_down = (rotation == 0) and (is_beginning or is_previous_rotation)
                total_frames = config["frames"]
                total_frames = total_frames + config["frames"] // 5 if speed_up else total_frames
                total_frames = total_frames + config["frames"] // 5 if speed_down else total_frames
                
                kf_interp = KeyframeInterp(config, inpainter_pipeline, None, vae, rotation, 
                                    ToPILImage()(kf1_image[0]), inpainting_prompt, adaptive_negative_prompt,
                                    kf2_upsample_coef=config['kf2_upsample_coef'], kf1_image=kf1_image, kf2_image=kf2_image,
                                    kf1_depth=kf1_depth, kf2_depth=kf2_depth, kf1_camera=kf1_camera, kf2_camera=kf2_camera, kf2_mask=kf2_mask,
                                    speed_up=speed_up, speed_down=speed_down, total_frames=total_frames
                                    ).to(config["device"])
                save_root = Path(kf_interp.run_dir) / "images"
                save_root.mkdir(exist_ok=True, parents=True)
                ToPILImage()(kf1_image[0]).save(save_root / "kf1.png")
                ToPILImage()(kf2_image[0]).save(save_root / "kf2.png")

                kf2_camera_upsample, kf2_depth_upsample, kf2_mask_upsample, kf2_image_upsample = kf_interp.upsample_kf2()

                kf_interp.update_additional_point_cloud(kf2_depth_upsample, kf2_image_upsample, valid_mask=kf2_mask_upsample, camera=kf2_camera_upsample, points_2d=kf_interp.points_kf2)
                inconsistent_additional_point_index = kf_interp.visibility_check()
                kf2_depth_updated = kf_interp.update_additional_point_depth(inconsistent_additional_point_index, depth=kf2_depth_upsample, mask=kf2_mask_upsample)
                
                kf_interp.reset_additional_point_cloud()
                kf_interp.update_additional_point_cloud(kf2_depth_updated, kf2_image_upsample, valid_mask=kf2_mask_upsample, camera=kf2_camera_upsample, points_2d=kf_interp.points_kf2)
                
                kf_interp.depths[0] = torch.nn.functional.interpolate(kf2_depth_updated, size=(512, 512), mode="nearest")
                self.evaluate_epoch(kf_interp, 0, vmax=vmax)

                for epoch in range(1, total_frames + 1):
                    render_output_kf1 = kf_interp.render_kf1(epoch)

                    inpaint_output = kf_interp.inpaint(render_output_kf1["rendered_image"], render_output_kf1["inpaint_mask"])

                    if config["finetune_decoder_interp"]:
                        finetune_decoder(config, kf_interp, render_output_kf1, inpaint_output, config["num_finetune_decoder_steps_interp"])

                    kf_interp.update_images_and_masks(inpaint_output["latent"], render_output_kf1["inpaint_mask"])

                    kf_interp.update_additional_point_cloud(render_output_kf1["rendered_depth"], kf_interp.images[-1], append_depth=True)

                    kf_interp.vae.decoder = copy.deepcopy(kf_interp.decoder_copy)
                    with torch.no_grad():
                        kf_interp.images_orig_decoder.append(kf_interp.decode_latents(inpaint_output["latent"]).detach())
                    self.evaluate_epoch(kf_interp, epoch, vmax=cutoff_depth*0.95)
                    self.empty_cache()

                kf_interp.images.append(kf1_image) 
                self.evaluate(kf_interp)

                all_rundir.append(kf_interp.run_dir)

        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = Path(config['runs_dir']) / f"{dt_string}_merged"
        if not config['skip_interp']:
            merge_frames(all_rundir, save_dir=save_dir, fps=config["save_fps"], is_forward=True, save_depth=False, save_gif=False)
        merge_keyframes(all_keyframes, save_dir=save_dir)
        pt_gen.write_all_content(save_dir=save_dir)

        return save_dir

    def __call__(self, operator):
        return self.process(operator)