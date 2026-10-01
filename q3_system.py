"""Homework 1, Question 3: solve Ax = b using the inverse matrix."""

import numpy as np


A = np.array([[2.0, 1.0], [3.0, -1.0]])
b = np.array([5.0, 4.0])

A_inverse = np.linalg.inv(A)
solution = A_inverse @ b

print("A inverse =\n", A_inverse)
print("\n[x, y] =", solution)
print("\nCheck A @ solution =", A @ solution)
