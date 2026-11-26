"""IrisMix viewer: an interactive web app for composing, rendering and editing makeup looks (see irismix/viewer/cli.py).

  python scripts/viewer.py                       # http://localhost:7007
  ssh -L 7007:localhost:7007 <user>@<host>       # from your computer, when the viewer runs on a remote GPU machine"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from irismix.viewer.cli import main                           # noqa: E402

if __name__ == '__main__':
    main()
