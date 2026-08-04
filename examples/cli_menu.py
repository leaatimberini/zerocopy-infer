#!/usr/bin/env python3
"""
ZeroCopy-Infer: Interactive Terminal Model Selector & Inference Launcher
========================================================================
Authored by Leandro Emanuel Timberini (Investigador Independiente — Ituzaingó, Buenos Aires, Argentina).
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from zerocopy_infer.cli import main

if __name__ == "__main__":
    main()
