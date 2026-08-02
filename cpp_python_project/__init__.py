"""Python package exposing the C++ core extension built with nanobind."""

import os
import sys

# On Windows, add package directory and system MinGW bin to DLL search path if needed
if sys.platform == "win32":
    pkg_dir = os.path.dirname(__file__)
    if os.path.exists(pkg_dir):
        try:
            os.add_dll_directory(pkg_dir)
        except Exception:
            pass
    mingw_bin = r"C:\Users\ajayl\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.MCF.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
    if os.path.exists(mingw_bin):
        try:
            os.add_dll_directory(mingw_bin)
        except Exception:
            pass

from . import core

__all__ = ["core"]
__version__ = "0.2.0"