import numpy as np


arr = np.zeros((80,80))

X, Y = 40, 79
a, b = 2, 8
Radius = 10
for x in range(80):
    for y in range(80):
        if (((X-x)**2)/a**2)+(((Y-y)**2)/b**2) > Radius**2:
            arr[y, x] = 1


print(arr)