"""Homework 1, Question 1: basic matrix operations."""

import numpy as np


A = np.array([[1, 2], [3, 4]], dtype=int)
B = np.array([[2, 0], [1, 2]], dtype=int)

print("A + B =\n", A + B)
print("\nA - B =\n", A - B)
print("\n2A =\n", 2 * A)
print("\nAB =\n", A @ B)
print("\nBA =\n", B @ A)
print("\nAB equals BA:", np.array_equal(A @ B, B @ A))
