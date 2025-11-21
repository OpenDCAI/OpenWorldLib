import torch


class BaseSynthesis(object):
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_path, args, device=None, **kwargs):
        pass

    @torch.no_grad()
    def predict(self):
        pass
