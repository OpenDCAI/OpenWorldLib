from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import sys

# Prefer the setup script-provided Recast path. The fallback matches the
# repository-level submodules layout.
recast_path = os.environ.get("RECAST_PATH", "../hunyuan-worldplay2-recastnavigation")
# 转换为绝对路径（避免 pip 安装时工作目录变化导致路径错误）
recast_path = os.path.abspath(recast_path)

# 路径检查
if not os.path.exists(recast_path):
    print(f"\033[31m错误: RecastNavigation 路径不存在: {recast_path}\033[0m")
    print("解决方法：")
    print("  1. 运行 scripts/setup/hunyuan_worldplay2_install.sh 自动下载")
    print("  2. 或通过环境变量指定路径：export RECAST_PATH=/path/to/recastnavigation")
    raise SystemExit(1)

# -------------------------- 2. 编译源文件配置（不变，仅路径为绝对） --------------------------
sources = [
    "full_recast_bindings.cpp",
    "navmesh_builder.cpp",
    # Recast 源文件（绝对路径，pip 安装时更可靠）
    os.path.join(recast_path, "Recast/Source/Recast.cpp"),
    os.path.join(recast_path, "Recast/Source/RecastAlloc.cpp"),
    os.path.join(recast_path, "Recast/Source/RecastArea.cpp"),
    os.path.join(recast_path, "Recast/Source/RecastAssert.cpp"),
    os.path.join(recast_path, "Recast/Source/RecastContour.cpp"),
    os.path.join(recast_path, "Recast/Source/RecastFilter.cpp"),
    os.path.join(recast_path, "Recast/Source/RecastLayers.cpp"),
    os.path.join(recast_path, "Recast/Source/RecastMesh.cpp"),
    os.path.join(recast_path, "Recast/Source/RecastMeshDetail.cpp"),
    os.path.join(recast_path, "Recast/Source/RecastRasterization.cpp"),
    os.path.join(recast_path, "Recast/Source/RecastRegion.cpp"),
    # Detour 源文件
    os.path.join(recast_path, "Detour/Source/DetourAlloc.cpp"),
    os.path.join(recast_path, "Detour/Source/DetourAssert.cpp"),
    os.path.join(recast_path, "Detour/Source/DetourCommon.cpp"),
    os.path.join(recast_path, "Detour/Source/DetourNavMesh.cpp"),
    os.path.join(recast_path, "Detour/Source/DetourNavMeshBuilder.cpp"),
    os.path.join(recast_path, "Detour/Source/DetourNavMeshQuery.cpp"),
    os.path.join(recast_path, "Detour/Source/DetourNode.cpp"),
]

# -------------------------- 3. 扩展模块配置（补充跨平台编译） --------------------------
ext_modules = [
    Pybind11Extension(
        "recast",  # 最终包名（import recast 对应这个名字）
        sources,
        include_dirs=[
            os.path.join(recast_path, "Recast/Include"),
            os.path.join(recast_path, "Detour/Include"),
            os.path.abspath("."),  # 当前目录绝对路径，避免 pip 临时目录问题
        ],
        define_macros=[
            ('RC_MAX_VERTS_PER_POLY', '6'),
        ],
        cxx_std=11,  # Recast 要求 C++11，强制指定避免编译错误
        # 跨平台编译优化（可选，不影响全局导入）
        extra_compile_args=["-O3"] if not sys.platform.startswith("win") else ["/O2"],
    ),
]

# -------------------------- pip 标准安装配置 --------------------------
setup(
    name="recast",  # 包名，pip install 时的名字
    version="0.1.0",  # 版本号
    author="ewrfcas&zhenyangliu",
    description="Pybind11 bindings for RecastNavigation",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},  # pybind11 编译必需（关键！）
    zip_safe=False,  # 扩展模块不能压缩，必须设为 False
    python_requires=">=3.6",  # 指定兼容的 Python 版本（可选，避免版本问题）
)
