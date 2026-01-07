import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        # 这里建议顺便加上 weights_only=False 消除之前的警告
        parameters = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        if "optimizer" in parameters:
            parameters = parameters["model"]
        # 关键修改：允许非严格匹配
        self.load_state_dict(parameters, strict=False) 
        print(f"[Depth Model] Successfully loaded weights from {path} with strict=False")

