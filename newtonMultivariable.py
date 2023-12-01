import numpy as np

def newton_multivariable(f, J, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for solving multivariable equations.
    
    Parameters:
        f (function): The system of equations as a function f(x) = 0.
        J (function): The Jacobian matrix of the system of equations.
        x0 (array-like): Initial guess for the solution.
        tol (float): Tolerance for convergence (default: 1e-6).
        max_iter (int): Maximum number of iterations (default: 100).
    
    Returns:
        x (array-like): The approximate solution.
        iter (int): The number of iterations taken.
    """
    x = np.array(x0, dtype=np.float64)
    for i in range(max_iter):
        delta_x = np.linalg.solve(J(x), -f(x))
        x += delta_x
        if np.linalg.norm(delta_x) < tol:
            return x, i+1
    return x, max_iter

# Example usage
def f(x):
    return np.array([
        x[0]**2 + x[1]**2 - 1,
        x[0] - x[1]**2
    ])

def J(x):
    return np.array([
        [2*x[0], 2*x[1]],
        [1, -2*x[1]]
    ])

x0 = [1, 1]
solution, iterations = newton_multivariable(f, J, x0)
print("Solution:", solution)
print("Iterations:", iterations)
