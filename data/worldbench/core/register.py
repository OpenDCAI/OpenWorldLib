class Registry:
    """
    通用注册器类，用于管理模型、任务、指标等模块的注册与实例化。
    实现原理：维护一个全局字典，将字符串名称映射到具体的类或函数。
    """
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._name}, items={list(self._module_dict.keys())})"

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """
        根据名称获取已注册的类
        Args:
            key (str): 注册时的名称（通常是类名）
        Returns:
            class: 对应的类
        """
        if key not in self._module_dict:
            raise KeyError(f"'{key}' is not registered in {self._name} registry! "
                           f"Available: {list(self._module_dict.keys())}")
        return self._module_dict[key]

    def register(self, module=None, name=None):
        """
        注册装饰器。
        
        Usage:
            1. @REGISTRY.register()
            class MyClass: ...
            
            2. @REGISTRY.register(name="MyCustomName")
            class MyClass: ...
        """
        # 内部包装函数，用于处理装饰器逻辑
        def _register(cls_or_func):
            # 确定注册用的 key：如果有指定 name 就用 name，否则用类名
            key = name if name is not None else cls_or_func.__name__
            
            if key in self._module_dict:
                print(f"[Warning] {key} is already registered in {self._name}, overwriting it.")
            
            self._module_dict[key] = cls_or_func
            return cls_or_func

        # 如果直接作为函数调用 registry.register(MyClass)
        if module is not None:
            return _register(module)

        # 如果作为装饰器调用 @registry.register()
        return _register

# 用于注册具体的模型实现
MODEL_REGISTRY = Registry("MODEL")

# 用于注册任务逻辑
TASK_REGISTRY = Registry("TASK")

# 用于注册评测指标
METRIC_REGISTRY = Registry("METRIC")