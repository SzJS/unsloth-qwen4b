#!/usr/bin/env python
"""Evaluate model locally — see evaluation.local for details."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from evaluation.local import main
main()
