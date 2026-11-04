import sys

from . import build

for p in build(sys.argv[1] if len(sys.argv) > 1 else "gcc"):
    print(p)
