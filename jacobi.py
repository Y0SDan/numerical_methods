import numpy as np

def jacobi(A, b, x0, max_iterations=100, tolerance=1e-6):
    """
    Implementa el método de Jacobi para resolver el sistema de ecuaciones lineales Ax = b.

    Parámetros:
    A: Matriz de coeficientes.
    b: Vector de términos independientes.
    x0: Vector inicial de la solución.
    max_iterations: Número máximo de iteraciones (por defecto 100).
    tolerance: Tolerancia para la convergencia (por defecto 1e-6).

    Devuelve:
    x: Vector de solución del sistema.
    """
    n = len(A)  # Número de filas (o columnas) en A
    x = x0.copy()  # Copia del vector inicial de la solución

    # Iterar hasta el número máximo de iteraciones
    for k in range(max_iterations):
        x_new = np.zeros_like(x)  # Vector para almacenar la nueva solución

        # Calcular la nueva solución
        for i in range(n):
            s = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s) / A[i, i]

        # Comprobar la convergencia
        if np.linalg.norm(x_new - x) < tolerance:
            return x_new  # Devolver la solución si se ha alcanzado la convergencia

        x = x_new  # Actualizar la solución

    return x  # Devolver la solución

# Ejemplo de uso
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])  # Matriz de coeficientes
b = np.array([5, 5, 10])  # Vector de términos independientes
x0 = np.zeros_like(b)  # Vector inicial de la solución
solution = jacobi(A, b, x0)  # Resolver el sistema
print(solution)  # Imprimir la solución