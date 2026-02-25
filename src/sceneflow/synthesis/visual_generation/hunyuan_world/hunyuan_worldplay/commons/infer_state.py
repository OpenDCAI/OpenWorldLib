from typing import Optional
from dataclasses import dataclass, field


@dataclass
class InferState:
    enable_sageattn: bool = False  # whether to use SageAttention
    sage_blocks_range: Optional[range] = None  # block range to use SageAttention
    enable_torch_compile: bool = False  # whether to use torch compile

    # fp8 gemm related
    use_fp8_gemm: bool = False  # whether to use fp8 gemm
    quant_type: str = "fp8-per-block"  # fp8 quantization type
    include_patterns: list = field(
        default_factory=lambda: ["double_blocks"]
    )  # include patterns for fp8 gemm

    # vae related
    use_vae_parallel: bool = False  # whether to use vae parallel


__infer_state = None


def parse_range(value):
    if "-" in value:
        start, end = map(int, value.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(x) for x in value.split(",")]


def initialize_infer_state(args=None, **kwargs):
    global __infer_state
    
    # If args is provided, use it as source; otherwise construct from kwargs
    if args is None:
        from types import SimpleNamespace
        args = SimpleNamespace(**kwargs)
        
    # Helper to safely get attributes with defaults (handling both object and dict-like behavior if needed)
    def get_arg(name, default=None):
        return getattr(args, name, default)

    sage_blocks_range_val = get_arg("sage_blocks_range", "0-53")
    sage_blocks_range = parse_range(sage_blocks_range_val) if sage_blocks_range_val else None
    
    # Map CLI argument use_sageattn to internal enable_sageattn field
    use_sageattn = get_arg("use_sageattn", False)

    # Parse include_patterns from args
    include_patterns = get_arg("include_patterns", "double_blocks")
    if isinstance(include_patterns, str):
        # Split by comma and strip whitespace
        include_patterns = [p.strip() for p in include_patterns.split(",") if p.strip()]

    __infer_state = InferState(
        enable_sageattn=use_sageattn,
        sage_blocks_range=sage_blocks_range,
        enable_torch_compile=get_arg("enable_torch_compile", False),
        # fp8 gemm related
        use_fp8_gemm=get_arg("use_fp8_gemm", False),
        quant_type=get_arg("quant_type", "fp8-per-block"),
        include_patterns=include_patterns,
        # vae related
        use_vae_parallel=get_arg("use_vae_parallel", False),
    )
    return __infer_state


def get_infer_state():
    return __infer_state
