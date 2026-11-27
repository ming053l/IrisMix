"""IrisMix inference from a checkout (the same as the installed command irismix-infer; see irismix/infer.py).

  python scripts/infer.py --src source.jpg --ref reference.jpg --out out/"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from irismix.infer import main                                 # noqa: E402

if __name__ == '__main__':
    main()
