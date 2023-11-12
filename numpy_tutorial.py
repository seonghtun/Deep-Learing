import numpy as np

x = np.array([1.0, 2.0, 3.0])
print(x)

print(type(x))

y = np.array([2.0, 4.0, 6.0])

print(x + y)
print(x * y)
print(x / y)
print(x - y)

X = np.array([[51,55],[22,33],[21,45]])

print(X.flatten())
print(X > 30)
print(X[X > 30])