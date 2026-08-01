from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

ext_modules = [
    Pybind11Extension(
        "cpp_python_project.core",
        ["src/core.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="cpp-python-project",
    version="0.1.0",
    author="Aidan Law",
    description="Python project with a C++ extension built via pybind11",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=["pybind11>=2.10"],
)