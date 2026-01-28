#!/usr/bin/env python
"""Evaluate model via API — see evaluation.api for details."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from evaluation.api import main
main()
