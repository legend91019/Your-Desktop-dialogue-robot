import os
import sys
from pathlib import Path

root = str(Path(__file__).parent.absolute())
#print(root)
print(root)
print(os.path.dirname(os.path.abspath(__file__)))