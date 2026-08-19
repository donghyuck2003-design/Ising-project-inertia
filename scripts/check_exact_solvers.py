#!/usr/bin/env python
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from oapi.exact_solvers import available_backends
if __name__=='__main__':
    for k,v in available_backends().items():
        print(f'{k:12s}: {"available" if v else "not installed/importable"}')
