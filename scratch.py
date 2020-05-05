import numpy as np

a = np.array([1, 3, 5, 6, 4, 5, 8, 9])

print(a.shape)

b = np.reshape(a, (-1, 2))
print(b)