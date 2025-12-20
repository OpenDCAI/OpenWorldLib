# models/base_model.py
from abc import ABC, abstractmethod
from core import MODEL_REGISTRY

class BaseModel(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" # 默认配置

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def generate(self, inputs: dict):
        pass