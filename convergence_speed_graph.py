import numpy as np
import matplotlib.pyplot as plt

class RootFinder:
    def __init__(self, f, df=None, g=None, x0=None, a=None, b=None, alpha=None, tol=1e-6, max_iter=100):
        self.f = f          
        self.df = df        
        self.g = g          
        self.x0 = x0        
        self.a = a          
        self.b = b          
        self.alpha = alpha  
        self.tol = tol
        self.max_iter = max_iter
        
    def bisection(self):
        if self.f(self.a) * self.f(self.b) >= 0:
            raise ValueError("Function must have opposite signs at interval endpoints")
        
        errors = []
        x_prev = self.a
        a, b = self.a, self.b
        
        for i in range(self.max_iter):
            c = (a + b) / 2
            errors.append(abs(c - x_prev))
            
            if abs(self.f(c)) < self.tol:
                return c, errors
            
            if self.f(c) * self.f(a) < 0:
                b = c
            else:
                a = c
            
            x_prev = c
            
        return c, errors
    
    def fixed_point(self):
        if self.g is None:
            self.g = lambda x: x + self.alpha * self.f(x)
        
        errors = []
        x = self.x0
        
        for i in range(self.max_iter):
            x_new = self.g(x)
            errors.append(abs(x_new - x))
            
            if abs(x_new - x) < self.tol:
                return x_new, errors
                
            x = x_new
            
        return x, errors
    
    def newton(self):
        if self.df is None:
            raise ValueError("Derivative function is required for Newton's method")
        
        errors = []
        x = self.x0
        
        for i in range(self.max_iter):
            if abs(self.df(x)) < 1e-10:  
                raise ValueError("Derivative too close to zero")
                
            x_new = x - self.f(x) / self.df(x)
            errors.append(abs(x_new - x))
            
            if abs(x_new - x) < self.tol:
                return x_new, errors
                
            x = x_new
            
        return x, errors

def test_case_1():
    f = lambda x: (x-1)**3
    df = lambda x: 3*(x-1)**2
    g = lambda x: x - 0.1*((x-1)**3)  
    
    finder = RootFinder(
        f=f,
        df=df,
        g=g,
        x0=2.0,       
        a=0.0,        
        b=2.0,
        alpha=0.1,    
        tol=1e-6,
        max_iter=100
    )
    
    results = {}
    try:
        x_bis, errors_bis = finder.bisection()
        results['Bisection'] = errors_bis
    except Exception as e:
        print(f"Bisection failed: {e}")
    
    try:
        x_fp, errors_fp = finder.fixed_point()
        results['Fixed Point'] = errors_fp
    except Exception as e:
        print(f"Fixed Point failed: {e}")
    
    try:
        x_newton, errors_newton = finder.newton()
        results["Newton's"] = errors_newton
    except Exception as e:
        print(f"Newton's method failed: {e}")
    
    plt.figure(figsize=(10, 6))
    for method, errors in results.items():
        plt.semilogy(range(len(errors)), errors, label=method)
    
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_case_1()