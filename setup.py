from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import sys
import os

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stderr.reconfigure(line_buffering=True)

def log(msg: str):
    print(msg)
    print(msg, file=sys.stderr, flush=True)

# 1) Arch policy:
#    - If user provides TORCH_CUDA_ARCH_LIST -> respect it
#    - Else set a safe default (edit to your taste)
DEFAULT_TORCH_CUDA_ARCH_LIST = "7.5;8.0;8.6;8.9;9.0;12.0+PTX"

if "TORCH_CUDA_ARCH_LIST" not in os.environ or not os.environ["TORCH_CUDA_ARCH_LIST"].strip():
    os.environ["TORCH_CUDA_ARCH_LIST"] = DEFAULT_TORCH_CUDA_ARCH_LIST
    log(f"[fused-bilagrid] TORCH_CUDA_ARCH_LIST not set; defaulting to: {os.environ['TORCH_CUDA_ARCH_LIST']}")
else:
    log(f"[fused-bilagrid] Using TORCH_CUDA_ARCH_LIST from env: {os.environ['TORCH_CUDA_ARCH_LIST']}")

# 2) Keep nvcc flags strictly "behavioral/optimization", NOT arch selection.
nvcc_args = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "--generate-line-info",
    "--source-in-ptx",
]

class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        log("\n" + "=" * 50)
        log(f"[fused-bilagrid] Building with TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST')}")
        log("=" * 50 + "\n")
        super().build_extensions()

setup(
    name="fused_bilagrid",
    packages=["fused_bilagrid"],
    ext_modules=[
        CUDAExtension(
            name="fused_bilagrid_cuda",
            sources=[
                "fused_bilagrid/sample_forward.cu",
                "fused_bilagrid/sample_backward.cu",
                "fused_bilagrid/uniform_sample.cu",
                "fused_bilagrid/tv_loss_forward.cu",
                "fused_bilagrid/tv_loss_backward.cu",
                "fused_bilagrid/ext.cpp",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": CustomBuildExtension},
)
