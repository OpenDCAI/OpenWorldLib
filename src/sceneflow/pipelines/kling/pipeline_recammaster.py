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
                        model_path="KlingTeam/ReCamMaster-Wan2.1",
                        required_components={"wan_model_path": "Wan-AI/Wan2.1-T2V-1.3B"},
                        device="cuda",
                        weight_dtype = torch.bfloat16,
                        **kwargs):
        if isinstance(required_components, dict) and "wan_model_path" in required_components.keys():
            wan_model_path = required_components.get("wan_model_path", "Wan-AI/Wan2.1-T2V-1.3B")
        else:
            wan_model_path = "Wan-AI/Wan2.1-T2V-1.3B"
        synthesis_model = ReCamMasterSynthesis.from_pretrained(pretrained_model_path=wan_model_path,
                                                         recammaster_ckpt_path=model_path,
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
                 camera_trajectory,
                 video_path,
                 prompt,
                 num_frames=81,
                 max_num_frames=81,
                 frame_interval=1,
                 size=(480, 832),
                 ):
        height, width = size
        self.operator.max_num_frames = max_num_frames
        self.operator.frame_interval = frame_interval
        self.operator.num_frames = num_frames
        self.operator.height = height
        self.operator.width = width

        video, cam_trajectory_emb, textual_prompt = self.process(camera_trajectory,
                                                                 video_path,
                                                                 prompt)
        
        output_video = self.synthesis_model.predict(
                                            textual_prompt,
                                            video,
                                            cam_trajectory_emb,
                                            num_frames=num_frames,
                                            height=height,
                                            width=width)
        return output_video
