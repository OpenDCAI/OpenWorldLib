from .navigation_video_generation import (
    reference_func as nav_video_gen_ref_func,
    eval_func as nav_video_gen_eval_func,
)
from .imagetext2video_generation import (
    reference_func as imagetext2video_gen_ref_func,
    eval_func as imagetext2video_gen_eval_func,
)


eval_func_mapping = {
    "navigation_video_gen": {
        "reference_func": nav_video_gen_ref_func,
        "eval_func": nav_video_gen_eval_func
    },
    "imagetext2video_gen": {
        "reference_func": imagetext2video_gen_ref_func,
        "eval_func": imagetext2video_gen_eval_func
    }
}
