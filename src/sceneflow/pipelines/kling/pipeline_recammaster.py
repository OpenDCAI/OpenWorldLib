import torch
from typing import Optional
from ...operators.recammaster_operator import ReCamMasterOperator
from ...synthesis.visual_generation.kling.recammaster_synthesis import ReCamMasterSynthesis


class ReCamMasterPipeline:
    def __init__(self,
                 operator: Optional[ReCamMasterOperator] = None,
                 synthesis_model: Optional[ReCamMasterSynthesis] = None,
                 device: str = "cuda",
                 weight_dtype = torch.bfloat16,):
        self.synthesis_model = synthesis_model 
        self.operator = operator
        self.device = device
        self.weight_dtype = weight_dtype

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_path="Wan-AI/Wan2.1-T2V-1.3B",
                        recammaster_ckpt_path="KlingTeam/ReCamMaster-Wan2.1",
                        device="cuda",
                        weight_dtype = torch.bfloat16,
                        **kwargs):
        synthesis_model = ReCamMasterSynthesis.from_pretrained(pretrained_model_path=pretrained_model_path,
                                                         recammaster_ckpt_path=recammaster_ckpt_path,
                                                         device=device,
                                                         weight_dtype=weight_dtype)
        operator = ReCamMasterOperator()
        return cls(operator, synthesis_model, device, weight_dtype)

    def process(self,
                interaction,
                video_path,
                textual_prompt):
        video = self.operator.process_perception(video_path).to(self.weight_dtype)

        self.operator.get_interaction(interaction, textual_prompt)
        cam_trajectory_emb = self.operator.process_interaction().to(self.weight_dtype)

        self.operator.delete_last_interaction()

        return video, cam_trajectory_emb, textual_prompt

    def __call__(self,
                 interaction,
                 video_path,
                 textual_prompt,
                 max_num_frames=81,
                 frame_interval=1,
                 num_frames=81,
                 height=480,
                 width=832
                 ):
        self.operator.max_num_frames = max_num_frames
        self.operator.frame_interval = frame_interval
        self.operator.num_frames = num_frames
        self.operator.height = height
        self.operator.width = width

        video, cam_trajectory_emb, textual_prompt = self.process(interaction,
                                                                 video_path,
                                                                 textual_prompt)
        
        output_video = self.synthesis_model.predict(
                                            textual_prompt,
                                            video,
                                            cam_trajectory_emb,
                                            num_frames=num_frames,
                                            height=height,
                                            width=width)
        return output_video
