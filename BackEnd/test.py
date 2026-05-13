import os
import sys
from pathlib import Path

root = str(Path(__file__).parent.absolute())
#print(root)
sys.path.append(root)

#print(sys.path)

print(os.path.join(os.path.dirname(__file__)))
print(os.path.join(os.path.dirname(__file__),'.'))
print(os.path.join(os.path.dirname(__file__),'..'))
print(os.path.join(os.path.dirname(__file__), '..', 'config.json'))
print(__file__)