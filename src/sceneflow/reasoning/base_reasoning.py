import torch


class BaseReasoning(object):
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_path, device=None, **kwargs):
        pass
    
    def api_init(self, api_key, endpoint):
        pass

    @torch.no_grad()
    def inference(self):
        pass