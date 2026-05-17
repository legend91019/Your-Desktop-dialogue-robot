import os
import sys
from pathlib import Path

print(os.path.abspath(__file__))
print(Path(__file__).absolute())

print(Path(__file__).parent.parent.absolute())
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(__name__)