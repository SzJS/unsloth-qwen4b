#!/usr/bin/env python
"""Evaluate base model with inoculation prefill — see evaluation.base_inoculation for details."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from evaluation.base_inoculation import main
main()
