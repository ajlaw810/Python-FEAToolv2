# Python-FEAToolv2

A Python project with a high-performance C++ extension built using [pybind11](https://github.com/pybind/pybind11).

This project is entirely separate from the `ajlaw810.github.io` website.

## Project Structure

```
cpp-python-project/
├── src/
│   └── core.cpp              # C++ source with pybind11 bindings
├── cpp_python_project/       # Python package
│   └── __init__.py           # Imports and exposes the C++ extension
├── tests/
│   └── test_core.py          # Tests for the C++ extension
├── setup.py                  # Build configuration (pybind11 + setuptools)
├── pyproject.toml            # Build-system requirements
├── requirements.txt          # Python dependencies
├── README.md
└── .gitignore
```

## Requirements

- Python 3.8+
- A C++ compiler (MSVC on Windows, GCC/Clang on Linux/macOS)
- CMake is not required; pybind11's setuptools helpers handle the build

## Setup & Build

```bash
# 1. (Recommended) Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build and install the C++ extension in editable mode
pip install -e .
```

## Usage

```python
from cpp_python_project import core

# Simple function
print(core.add(2, 3))          # 5

# Performance-oriented function
print(core.fibonacci(20))      # 6765

# Vector processing
print(core.scale_vector([1.0, 2.0, 3.0], 2.0))  # [2.0, 4.0, 6.0]

# Class with state
counter = core.Counter(5)
counter.increment(4)
print(counter.value())         # 9
```

## Running Tests

```bash
pytest
```

## License

This project is for personal/educational use.