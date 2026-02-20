from .generation import *
from .vla import *


tasks_map = {
    "navigation_video_gen": nav_videogen_benchmarks,
    "text2video_gen": text2video_benchmarks,
    "vla_evaluation": vla_libero_benchmarks,
}
