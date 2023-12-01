import numpy as np

def factorizacion_lu(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i][i] = 1

        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))

        for j in range(i + 1, n):
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return L, U

# Prueba del algoritmo
A = np.array([[2, -1, 3], [4, 2, -1], [-2, 3, 2]])
L, U = factorizacion_lu(A)

print("Matriz A:")
print(A)

print("Matriz L:")
print(L)

print("Matriz U:")
print(U)
