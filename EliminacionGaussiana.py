import numpy as np

def gaussian_elimination(A, b):
    n = len(A)
    for i in range(n):
        # Partial pivoting
        max_row = i
        for j in range(i+1, n):
            if abs(A[j][i]) > abs(A[max_row][i]):
                max_row = j
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        
        # Elimination
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            A[j][i] = 0
            for k in range(i+1, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]
    
    return x

# Definir la matriz A y el vector b
A = np.array([[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]], float)
b = np.array([1, -2, 0], float)

# Llamar a la función gaussian_elimination
x = gaussian_elimination(A, b)

# Imprimir el resultado
#print("hello world")
print(x)