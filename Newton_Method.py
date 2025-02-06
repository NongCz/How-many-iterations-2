import math

def newtons_method(f, df, x0, tol=1e-6, max_iterations=1000):
    x = x0
    for i in range(max_iterations):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            print("Derivative is zero. Method fails.")
            return None, i
        
        x_next = x - fx / dfx
        if abs(x_next - x) < tol:
            return x_next, i + 1  
        
        x = x_next

    return None, max_iterations  

experiments = [
    {
        "f": lambda x: x**2 - 2, 
        "df": lambda x: 2*x, 
        "x0": 1.0, 
        "description": "Solving x^2 - 2 = 0 (√2)"
    },
    {
        "f": lambda x: math.exp(x) - 3*x**2, 
        "df": lambda x: math.exp(x) - 6*x, 
        "x0": 1.0, 
        "description": "Solving e^x - 3x^2 = 0"
    },
    {
        "f": lambda x: math.sin(x) - 0.5, 
        "df": lambda x: math.cos(x), 
        "x0": 1.0, 
        "description": "Solving sin(x) - 0.5 = 0"
    },
]

tolerance = 1e-6  
for exp in experiments:
    root, iterations = newtons_method(exp["f"], exp["df"], exp["x0"], tol=tolerance)
    print(f"Function: {exp['description']}")
    print(f"  Initial guess: {exp['x0']}")
    print(f"  Root approximation: {root}")
    print(f"  Iterations needed: {iterations}")
    print("-" * 50)
