import numpy as np
import matplotlib.pyplot as plt
import math

# Function and its derivative
def f(x):
    return x**3 - x - 2  # Equation: x^3 - x - 2 = 0

def df(x):
    return 3*x**2 - 1  # Derivative: 3x^2 - 1

# Bisection Method
def bisection_method(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) >= 0:
        print("Bisection method fails. f(a) and f(b) must have opposite signs.")
        return None, []

    iterates = []
    for _ in range(max_iter):
        c = (a + b) / 2
        iterates.append(c)

        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c, iterates

        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return None, iterates

# Fixed Point Iteration
def fixed_point_iteration(g, x0, tol=1e-6, max_iter=100):
    iterates = [x0]
    for _ in range(max_iter):
        x_next = g(iterates[-1])
        iterates.append(x_next)

        if abs(x_next - iterates[-2]) < tol:
            return x_next, iterates
    return None, iterates

# Newton's Method
def newtons_method(f, df, x0, tol=1e-6, max_iter=100):
    iterates = [x0]
    for _ in range(max_iter):
        x_next = iterates[-1] - f(iterates[-1]) / df(iterates[-1])
        iterates.append(x_next)

        if abs(x_next - iterates[-2]) < tol:
            return x_next, iterates
    return None, iterates

# Define g(x) for fixed point iteration: x = g(x) --> g(x) = (x + 2/x)/2
def g(x):
    return (x + 2 / x) / 2

# Initial guesses
a, b = 1, 2  # Interval for Bisection Method
x0 = 1.5  # Initial guess for Fixed Point and Newton's Method

# Run the methods
bisection_root, bisection_iters = bisection_method(f, a, b)
fixed_point_root, fixed_point_iters = fixed_point_iteration(g, x0)
newton_root, newton_iters = newtons_method(f, df, x0)

# Plotting convergence
plt.figure(figsize=(10, 6))

plt.plot(range(len(bisection_iters)), bisection_iters, 'ro-', label="Bisection Method")
plt.plot(range(len(fixed_point_iters)), fixed_point_iters, 'go-', label="Fixed Point Iteration")
plt.plot(range(len(newton_iters)), newton_iters, 'bo-', label="Newton's Method")

plt.xlabel("Iteration")
plt.ylabel("Approximate Root")
plt.title("Comparison of Root-Finding Methods")
plt.legend()
plt.grid()
plt.show()