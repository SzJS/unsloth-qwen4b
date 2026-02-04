#!/usr/bin/env python
"""Compute perplexity of inoculation prefills — see analysis.perplexity for details."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from analysis.perplexity import main
if __name__ == "__main__":
    main()
