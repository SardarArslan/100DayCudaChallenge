import sys
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Force build location to current directory
sys.argv += ["build_ext", "--inplace"]

setup(
    name='addition_extension',
    ext_modules=[
        CUDAExtension(
            name='addition_extension',
            sources=['vecAdd1d.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)