import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import apex
import fused_adam_cuda, amp_C

setup(
    name='distopt',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='_distopt',
            include_dirs=[os.path.dirname(os.path.realpath(__file__))],
            sources=['csrc/dist_opt.cpp'],
            extra_objects=[fused_adam_cuda.__file__, amp_C.__file__],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    test_suite="tests",
)
