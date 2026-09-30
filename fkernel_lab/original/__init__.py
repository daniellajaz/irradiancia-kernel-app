"""Clases originales del repositorio https://github.com/magohector/fkernel
(carpeta Experimentos, licencia GPL, autor Héctor Andrés Mora Paz).

Los archivos KSVM.py y KANN.py hacen ``from KernelUtilities import *``, así que
se añade esta carpeta al ``sys.path`` antes de importarlos. Si faltan, se
descargan desde GitHub.
"""
import os
import sys
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = "https://raw.githubusercontent.com/magohector/fkernel/master/Experimentos/"
for _name in ("KernelUtilities.py", "KSVM.py", "KANN.py"):
    _path = os.path.join(_HERE, _name)
    if not os.path.exists(_path):
        urllib.request.urlretrieve(_BASE + _name, _path)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import KernelUtilities  # noqa: E402,F401
from KSVM import KSVC  # noqa: E402
from KANN import KANNC  # noqa: E402

__all__ = ["KSVC", "KANNC", "KernelUtilities"]
