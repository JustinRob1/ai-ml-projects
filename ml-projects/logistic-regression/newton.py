from math import sqrt

def newton_method(f, f_prime, x0, tolerance=1e-30, max_iterations=6):
    """
    Newton's method for finding the root of a function.

    Parameters:
    - f: The function for which to find the root.
    - f_prime: The derivative of the function.
    - x0: Initial guess for the root.
    - tolerance: Convergence tolerance.
    - max_iterations: Maximum number of iterations.

    Returns:
    - root: Approximation of the root.
    - iterations: Number of iterations performed.
    """

    x = x0
    iterations = 0

    print(f"Iteration 0: x = {x0:.50f}")

    while abs(f(x)) > tolerance and iterations < max_iterations:
        print("f(x) =", f(x))
        print("f'(x) =", f_prime(x))
        x = x - f(x) / f_prime(x)
        iterations += 1
        print(f"Iteration {iterations}: x = {x:.50f}")
        print()

    return x, iterations

def f(x):
    return 4*(x**5) + 10*(x**4) + sqrt(24)*(x**3) + x

def f_prime(x):
    return 20*(x**4) + 40*(x**3) + 3*sqrt(24)*(x**2) + 1

initial_guess = -1.0  # Initial guess close to the root
root, num_iterations = newton_method(f, f_prime, initial_guess)

print(f"\nRoot found: {root:.50f}")
print(f"Iterations performed: {num_iterations}")
