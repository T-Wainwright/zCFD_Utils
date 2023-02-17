from setuptools import setup, find_packages
from pybind11.setup_helpers import intree_extensions, build_ext
import platform

ext_modules = intree_extensions("src/boost/multiscale.cpp")
ext_modules[0].cxx_std = 17

if platform.system() != 'Darwin':
    # Need openmp on the linker for it to be able to use omp_get_num_threads()
    ext_modules[0].extra_link_args.append("-fopenmp")
    ext_modules[1].extra_link_args.append("-fopenmp")

ext_modules[0].extra_compile_args.append("-O3")
ext_modules[1].extra_compile_args.append("-O3")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="zcfdutils",
    version="0.0.2",
    author="Tom Wainwright",
    author_email="tom.wainwright@zenotech.com",
    description="Utilities for working with zCFD solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
