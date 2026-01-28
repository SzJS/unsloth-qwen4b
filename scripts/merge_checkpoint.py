#!/usr/bin/env python
"""Merge LoRA checkpoint into full model — see core.merge_checkpoint for details."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from core.merge_checkpoint import main
main()
