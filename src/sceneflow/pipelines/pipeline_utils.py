
class PipelineABC:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls):
        return cls()
    
    def __call__(self, *args, **kwds):
        pass

    def save_pretrained(self, save_directory: str):
        """
        finish this part after the training pipeline is prepared.
        """
        pass
