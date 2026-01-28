#!/usr/bin/env python
"""Train SFT — see training.sft for details."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from training.sft import main
main()
