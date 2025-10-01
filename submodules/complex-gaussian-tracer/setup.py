from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="complex-gaussian-tracer",
    version='0.1',
    packages=['complex_gaussian_tracer'],
    ext_modules=[
        CUDAExtension(
            name="complex_gaussian_tracer._C",
            sources=[
                "cuda_tracer/tracer_impl.cu",
                "cuda_tracer/forward.cu",
                "cuda_tracer/backward.cu",
                "tracer_points.cu",
                "ext.cpp"
            ],
            extra_compile_args={
                "nvcc": ["-I" + os.path.join(this_dir, "third_party/glm/")]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
