"""Homework 1, Question 4: determinant and rank."""

import numpy as np


C = np.array([[1.0, 2.0, 3.0],
              [1.0, 1.0, 1.0],
              [3.0, 5.0, 7.0]])

determinant = np.linalg.det(C)
rank = np.linalg.matrix_rank(C)

print("det(C) =", round(determinant))
print("rank(C) =", rank)
print("Row 3 equals 2*Row 1 + Row 2:",
      np.allclose(C[2], 2 * C[0] + C[1]))
