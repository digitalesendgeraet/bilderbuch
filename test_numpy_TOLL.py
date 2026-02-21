import numpy as np

weights = np.array([[[[2,1],[3,4]], [[2,1],[3,4]]], [[[2,1],[3,4]], [[2,1],[3,4]]]])
values = np.array([[2,1],[3,4]])


z = np.matmul(values, weights).sum(axis=(0,1))

sum = 0
for n in range(2):
    for m in range(2):
        sum += np.dot(values, weights[n,m])

print(z)
print(sum)