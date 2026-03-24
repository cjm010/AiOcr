from pathlib import Path
import sys


# Keep the repository root on sys.path so imports like `src.doc_ai...` work
# the same way locally and in GitHub Actions.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
