#!/usr/bin/env python
"""Evaluate LoRA checkpoint directly — see evaluation.eval_lora for details."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from evaluation.eval_lora import main
if __name__ == "__main__":
    main()
