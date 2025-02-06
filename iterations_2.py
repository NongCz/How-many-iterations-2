import random as r
import math

prev_x = r.uniform(1, 10)
eps = 0.05

def f(x):
    return -(x**2 - 2*x)/10 + x

i = 0
while True:
    print(f"{i}-th iteration: x = {prev_x}")
    x = f(prev_x)
    if abs(x - prev_x) < eps:
        break
    i += 1
    prev_x = x

print(f"It took {i} iterations to find root in precision of {eps}")