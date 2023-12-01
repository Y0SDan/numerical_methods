import numpy as np

def jacobi(A, b, x0, max_iterations=100, tolerance=1e-6):
    n = len(A)
    x = x0.copy()
    for k in range(max_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s) / A[i, i]
        if np.linalg.norm(x_new - x) < tolerance:
            return x_new
        x = x_new
    return x

# Example usage
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
b = np.array([5, 5, 10])
x0 = np.zeros_like(b)
solution = jacobi(A, b, x0)
print(solution)
