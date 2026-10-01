"""Homework 1, Question 5: inverse of a 3x3 matrix."""

import numpy as np


D = np.array([[2.0, 1.0, 0.0],
              [1.0, 1.0, 1.0],
              [0.0, 1.0, 1.0]])

D_inverse = np.linalg.inv(D)

print("D inverse =\n", D_inverse)
print("\nCheck D @ D_inverse =\n", np.round(D @ D_inverse, 10))
