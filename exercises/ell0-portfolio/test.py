from casadi import *
import numpy as np

# Create scalar/matrix symbols
x = MX.sym('x',5)

A = np.random.rand(5, 5)

# Compose into expressions
y = x.T @ A @ x

# Sensitivity of expression -> new expression
grad_y = gradient(y,x);

# Create a Function to evaluate expression
f = Function('f',[x],[grad_y])

# Evaluate numerically
grad_y_num = f([1,2,3,4,5]);

print(grad_y_num)