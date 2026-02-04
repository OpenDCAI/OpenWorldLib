"""
pipeline template:
```python
class PipelineABC:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls):
        ###### Load different categories of pretrained models here ######
        return cls()

    def process(self, *args, **kwds):
        ###### Process interaction signals using operators here ######
        pass

    def __call__(self, *args, **kwds):
        ###### This is the main interface called by the test file.
        ###### It should internally invoke the process() function.
        pass

    def stream(self, *args, **kwds) -> Generator[torch.Tensor, List[str], None]:
        ###### This function supports multi-round interactive inputs.
        ###### It should call __call__ internally.
        ###### Memory management must be handled here via the Memory module.
        pass
```
"""