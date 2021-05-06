
import numpy as np
from time import time

a = np.random.randn(32,32,3)

t = time()

#f1
s = ''
for i in a:
    for j in i:
        for x in j:
            s += str(x)
y = time()

''.join([[[str(x) for x in j] for j in i] for i in a])

now = time()
tot = now - y + y - t
print((y -t)/tot, (now-y)/tot)
