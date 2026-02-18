import sys
import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

# OpenMP flags depend on the compiler
c_args = []
l_args = []

if sys.platform == "win32":
    c_args = ['/openmp', '/O2']
else:
    c_args = ['-fopenmp', '-O3', '-march=native']
    l_args = ['-fopenmp']

ext_modules = [
    Pybind11Extension(
        "batch_grid_env",
        ["src/batched_env.cpp"],
        extra_compile_args=c_args,
        extra_link_args=l_args,
    ),
]

setuptools.setup(
    name="batch_grid_env",
    version="0.1",
    author="Your Name",
    description="High-performance batched multi-agent environment",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)