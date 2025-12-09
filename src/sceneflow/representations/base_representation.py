
class BaseRepresentation(object):
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_path, device=None, **kwargs):
        pass
    
    def api_init(self, api_key, endpoint):
        pass

    def get_representation(self, data):
        pass
