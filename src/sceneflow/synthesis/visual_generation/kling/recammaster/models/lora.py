import torch
from ......base_models.diffusion_model.video.wan_2p1.modules.model import WanModel


class GeneralLoRAFromPeft:
    def __init__(self):
        self.supported_model_classes = [WanModel]

    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_B." not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index("lora_B") + 2:
                keys.pop(keys.index("lora_B") + 1)
            keys.pop(keys.index("lora_B"))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
        return lora_name_dict

    def match(self, model: torch.nn.Module, state_dict_lora):
        lora_name_dict = self.get_name_dict(state_dict_lora)
        model_name_dict = {name: None for name, _ in model.named_parameters()}
        matched_num = sum([i in model_name_dict for i in lora_name_dict])
        if matched_num == len(lora_name_dict):
            return "", ""
        else:
            return None

    def fetch_device_and_dtype(self, state_dict):
        device, dtype = None, None
        for name, param in state_dict.items():
            device, dtype = param.device, param.dtype
            break
        computation_device = device
        computation_dtype = dtype
        if computation_device == torch.device("cpu"):
            if torch.cuda.is_available():
                computation_device = torch.device("cuda")
        if computation_dtype == torch.float8_e4m3fn:
            computation_dtype = torch.float32
        return device, dtype, computation_device, computation_dtype

    def load(self, model, state_dict_lora, lora_prefix="", alpha=1.0, model_resource=""):
        state_dict_model = model.state_dict()
        device, dtype, computation_device, computation_dtype = self.fetch_device_and_dtype(state_dict_model)
        lora_name_dict = self.get_name_dict(state_dict_lora)
        for name in lora_name_dict:
            weight_up = state_dict_lora[lora_name_dict[name][0]].to(device=computation_device, dtype=computation_dtype)
            weight_down = state_dict_lora[lora_name_dict[name][1]].to(device=computation_device, dtype=computation_dtype)
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2)
                weight_down = weight_down.squeeze(3).squeeze(2)
                weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_lora = alpha * torch.mm(weight_up, weight_down)
            weight_model = state_dict_model[name].to(device=computation_device, dtype=computation_dtype)
            weight_patched = weight_model + weight_lora
            state_dict_model[name] = weight_patched.to(device=device, dtype=dtype)
        print(f"    {len(lora_name_dict)} tensors are updated.")
        model.load_state_dict(state_dict_model)


def get_lora_loaders():
    return [GeneralLoRAFromPeft()]
