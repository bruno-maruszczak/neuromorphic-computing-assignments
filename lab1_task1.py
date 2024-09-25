import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Regulator model
def f(X, Y, params, i):
    a1, a2, a3, a4, a5 = params

    y =   a1 * X[i]\
        + a2 * (X[i - 1] if i - 1 >= 0 else 0)\
        + a3 * (X[i - 2] if i - 2 >= 0 else 0)\
        - a4 * (Y[i - 1] if i - 1 >= 0 else 0)\
        - a5 * (Y[i - 2] if i - 2 >= 0 else 0)

    
    return y


def J(params, X, Y):
    Y_pred = np.zeros_like(Y)
    for i in range(len(Y)):
        Y_pred[i] = f(X, Y_pred, params, i)
    return np.mean((Y - Y_pred) ** 2)

def optimize_params(X, Y, initial_params):
    # Minimize the MSE
    result = minimize(J, initial_params, args=(X, Y))
    return result.x, result.fun


Y = np.array([
        0.001, 1.18, 2.35, 2.62, 2.72, 2.65, 2.53, 2.42, 
        2.39, 2.39, 2.42, 2.45, 2.48, 2.467, 2.45, 
        2.44, 2.44, 2.455, 2.45, 2.449, 2.449, 2.447
    ])
X = [(0.0 if i == 0 else 1.0) for i in range(len(Y))]


initial_params = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Example initial parameters
optimized_params, min_mse = optimize_params(X, Y, initial_params)



# Results
print(f"Optimized Parameters: {optimized_params}")
print(f"Minimum Mean Square Error: {min_mse}")

Y_pred = np.zeros_like(Y)
for i in range(len(Y)):
    Y_pred[i] = f(X, Y_pred, optimized_params, i)

plt.figure(figsize=(10, 6))
plt.plot(Y, 'o-', label='Measured Output (Y)', color='blue')
plt.plot(Y_pred, 'x--', label='Predicted Output (Y\')', color='red')
plt.title('Measured Output vs Predicted Output')
plt.xlabel('Time Step')
plt.ylabel('Output Value')
plt.legend(loc='lower right')
plt.grid()
plt.show()