#!/usr/bin/env python
"""Train GRPO — see training.grpo for details."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from training.grpo import main
main()
