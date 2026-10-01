# Homework 1 — Matrix Algebra

**Student:** Xinchen Leng<br>
**Date:** September 29, 2026

This page presents the complete manual solutions and required NumPy verification code. The formatted submission is available as [Homework_1_Solutions.pdf](./Homework_1_Solutions.pdf).

---

## Part A. Conceptual / Manual Problems

## 1. Basic Operations

Given

$$
A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\qquad
B=\begin{bmatrix}2&0\\1&2\end{bmatrix}.
$$

### (a) Addition and subtraction

$$
A+B=\begin{bmatrix}1+2&2+0\\3+1&4+2\end{bmatrix}
=\boxed{\begin{bmatrix}3&2\\4&6\end{bmatrix}}
$$

$$
A-B=\begin{bmatrix}1-2&2-0\\3-1&4-2\end{bmatrix}
=\boxed{\begin{bmatrix}-1&2\\2&2\end{bmatrix}}
$$

### (b) Scalar multiplication

$$
2A=\boxed{\begin{bmatrix}2&4\\6&8\end{bmatrix}}
$$

### (c) Matrix products

$$
AB=\begin{bmatrix}1(2)+2(1)&1(0)+2(2)\\3(2)+4(1)&3(0)+4(2)\end{bmatrix}
=\boxed{\begin{bmatrix}4&4\\10&8\end{bmatrix}}
$$

$$
BA=\begin{bmatrix}2(1)+0(3)&2(2)+0(4)\\1(1)+2(3)&1(2)+2(4)\end{bmatrix}
=\boxed{\begin{bmatrix}2&4\\7&10\end{bmatrix}}
$$

Therefore, $AB\ne BA$. Matrix multiplication is generally not commutative: changing the order changes which rows are combined with which columns.

### NumPy verification

```python
import numpy as np

A = np.array([[1, 2], [3, 4]], dtype=int)
B = np.array([[2, 0], [1, 2]], dtype=int)

print("A + B =\n", A + B)
print("\nA - B =\n", A - B)
print("\n2A =\n", 2 * A)
print("\nAB =\n", A @ B)
print("\nBA =\n", B @ A)
print("\nAB equals BA:", np.array_equal(A @ B, B @ A))
```

```text
A + B =
 [[3 2]
  [4 6]]
A - B =
 [[-1  2]
  [ 2  2]]
2A =
 [[2 4]
  [6 8]]
AB =
 [[ 4  4]
  [10  8]]
BA =
 [[ 2  4]
  [ 7 10]]
AB equals BA: False
```

---

## 2. Properties of Special Matrices

| Matrix type | Example | Defining property |
|---|---|---|
| Diagonal | $\begin{bmatrix}3&0&0\\0&-2&0\\0&0&5\end{bmatrix}$ | All off-diagonal entries are zero. |
| Identity | $\begin{bmatrix}1&0\\0&1\end{bmatrix}$ | $IA=AI=A$. |
| Symmetric | $\begin{bmatrix}2&-1\\-1&4\end{bmatrix}$ | $S^T=S$. |
| Idempotent | $\begin{bmatrix}1&0\\0&0\end{bmatrix}$ | $P^2=P$. |

---

## 3. System of Equations

The system $2x+y=5$ and $3x-y=4$ has matrix form

$$
\underbrace{\begin{bmatrix}2&1\\3&-1\end{bmatrix}}_{A}
\underbrace{\begin{bmatrix}x\\y\end{bmatrix}}_{\mathbf{x}}
=\underbrace{\begin{bmatrix}5\\4\end{bmatrix}}_{\mathbf{b}}.
$$

Since $\det(A)=2(-1)-1(3)=-5\ne0$,

$$
A^{-1}=\frac{1}{-5}\begin{bmatrix}-1&-1\\-3&2\end{bmatrix}
=\begin{bmatrix}\frac15&\frac15\\\frac35&-\frac25\end{bmatrix}.
$$

Therefore,

$$
\begin{bmatrix}x\\y\end{bmatrix}
=A^{-1}\mathbf b
=\begin{bmatrix}\frac15&\frac15\\\frac35&-\frac25\end{bmatrix}
\begin{bmatrix}5\\4\end{bmatrix}
=\boxed{\begin{bmatrix}\frac95\\\frac75\end{bmatrix}},
$$

so $\boxed{(x,y)=(1.8,1.4)}$.

### NumPy verification

```python
import numpy as np

A = np.array([[2.0, 1.0], [3.0, -1.0]])
b = np.array([5.0, 4.0])

A_inverse = np.linalg.inv(A)
solution = A_inverse @ b

print("A inverse =\n", A_inverse)
print("\n[x, y] =", solution)
print("\nCheck A @ solution =", A @ solution)
```

```text
A inverse =
 [[ 0.2  0.2]
  [ 0.6 -0.4]]
[x, y] = [1.8 1.4]
Check A @ solution = [5. 4.]
```

---

## 4. Determinant and Rank

Given

$$
C=\begin{bmatrix}1&2&3\\1&1&1\\3&5&7\end{bmatrix},
$$

cofactor expansion along the first row gives

$$
\begin{aligned}
\det(C)
&=1\begin{vmatrix}1&1\\5&7\end{vmatrix}
-2\begin{vmatrix}1&1\\3&7\end{vmatrix}
+3\begin{vmatrix}1&1\\3&5\end{vmatrix}\\
&=(7-5)-2(7-3)+3(5-3)=2-8+6=\boxed{0}.
\end{aligned}
$$

Because $R_3=2R_1+R_2$, the three rows are linearly dependent. The first two rows are independent, so

$$
\boxed{\operatorname{rank}(C)=2}.
$$

The rank tells us that only two rows (and two columns) provide independent information. The matrix maps vectors into a two-dimensional subspace and is singular, consistent with its zero determinant.

### NumPy verification

```python
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
```

```text
det(C) = 0
rank(C) = 2
Row 3 equals 2*Row 1 + Row 2: True
```

---

## 5. Matrix Inverse

Given

$$
D=\begin{bmatrix}2&1&0\\1&1&1\\0&1&1\end{bmatrix},
$$

$\det(D)=-1$, so $D$ is invertible. Row reduction of $[D\mid I]$ (or the adjugate formula) gives

$$
\boxed{D^{-1}=\begin{bmatrix}0&1&-1\\1&-2&2\\-1&2&-1\end{bmatrix}}.
$$

Indeed,

$$
DD^{-1}=\begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix}=I_3.
$$

### NumPy verification

```python
import numpy as np

D = np.array([[2.0, 1.0, 0.0],
              [1.0, 1.0, 1.0],
              [0.0, 1.0, 1.0]])

D_inverse = np.linalg.inv(D)

print("D inverse =\n", D_inverse)
print("\nCheck D @ D_inverse =\n",
      np.round(D @ D_inverse, 10))
```

```text
D inverse =
 [[ 0.  1. -1.]
  [ 1. -2.  2.]
  [-1.  2. -1.]]
Check D @ D_inverse =
 [[1. 0. 0.]
  [0. 1. 0.]
  [0. 0. 1.]]
```

---

## Run All Python Checks

```bash
python3 -m pip install -r requirements.txt
python3 run_all.py
```

## Submission

- [Download the formatted PDF](./Homework_1_Solutions.pdf)
- [View the LaTeX source](./Homework_1_Solutions.tex)
- Individual `.py` files remain available for execution, but all answers, code, and outputs are displayed directly on this page.
