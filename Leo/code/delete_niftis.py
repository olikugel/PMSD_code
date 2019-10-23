import sys
import os

file = sys.argv[1]
with open(file) as f:
    for line in f:
        os.remove(line.rstrip('\n'))