from .generation import *
from .vla import *


tasks_map = {
    "navigation_video_gen": nav_videogen_benchmarks,
    "imagetext2video_gen": imagetext2video_benchmarks,
    "vla_eval": vla_libero_benchmarks,
}
