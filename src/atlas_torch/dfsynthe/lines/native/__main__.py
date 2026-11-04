import sys

from . import build

print(build(sys.argv[1] if len(sys.argv) > 1 else "gcc"))
