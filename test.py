import os
import sys
from pathlib import Path

root = str(Path(__file__).parent.absolute())
print(root)
sys.path.append(root)

print(sys.path)