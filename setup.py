from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mar_projector',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='mar_projector_backend',
            sources=[
                'src/csrc/projector.cpp',
                'src/csrc/projector_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-gencode=arch=compute_50,code=sm_50']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
