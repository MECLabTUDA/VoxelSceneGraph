"""File used to build the C++ layers."""

import warnings
from pathlib import Path

import torch
from setuptools import setup
# noinspection PyProtectedMember
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

if CUDA_HOME is None:
    warnings.warn("CUDA_HOME environment variable is not set")

# noinspection DuplicatedCode
csrc_dir = Path(__file__).absolute().parent.parent.parent / "csrc"

main_file = list(csrc_dir.glob("*.cpp"))
source_cpu = list((csrc_dir / "cpu").glob("*.cpp"))
source_cuda = list((csrc_dir / "cuda").glob("*.cu"))

source = main_file + source_cpu

extra_cflags = ["-I" + csrc_dir.as_posix()]

if torch.cuda.is_available() and CUDA_HOME is not None:
    source.extend(source_cuda)
    extra_cflags += ["-DWITH_CUDA"]
    ext_type = CUDAExtension
else:
    ext_type = CppExtension
source = list(map(lambda p: p.as_posix(), source))

# noinspection PyTypeChecker
setup(
    name="c_layers",
    ext_modules=[
        ext_type("c_layers", source, extra_compile_args={"cxx": extra_cflags}),
    ],
    cmdclass={"build_ext": BuildExtension}
)
