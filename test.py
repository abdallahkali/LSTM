import numpy as np 

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Input sequence
X = [1, 2, 3]

# LSTM weights (scalars)
W_f = 0.5; U_f = 0.1; b_f = 0
W_i = 0.6; U_i = 0.2; b_i = 0
W_c = 0.7; U_c = 0.3; b_c = 0
W_o = 0.8; U_o = 0.4; b_o = 0

# Output weights 
W_y = np.array([0.5, 0.5, 0.5, 0.5])
b_y = 0

# Initial states
h = 0
C = 0

print("=" * 50)
print("LSTM Numerical Example")
print("=" * 50)

#  time step
for t, x in enumerate(X, 1):
    print(f"\nTime Step t={t}, x={x}")
    print("-" * 30)
    
    f = sigmoid(W_f * x + U_f * h + b_f)
    print(f"f_{t} = {f:.4f}")
    
    i = sigmoid(W_i * x + U_i * h + b_i)
    print(f"i_{t} = {i:.4f}")
    
    c_tilde = tanh(W_c * x + U_c * h + b_c)
    print(f"c̃_{t} = {c_tilde:.4f}")
    
    C = f * C + i * c_tilde
    print(f"C_{t} = {C:.4f}")
    
    o = sigmoid(W_o * x + U_o * h + b_o)
    print(f"o_{t} = {o:.4f}")
    
    h = o * tanh(C)
    print(f"h_{t} = {h:.4f}")

# Predict next value
print("\n" + "=" * 50)
print(f"Prediction for x_4 = 4")
print("=" * 50)

f_4 = sigmoid(W_f * 4 + U_f * h + b_f)
i_4 = sigmoid(W_i * 4 + U_i * h + b_i)
c_tilde_4 = tanh(W_c * 4 + U_c * h + b_c)
C_4 = f_4 * C + i_4 * c_tilde_4
o_4 = sigmoid(W_o * 4 + U_o * h + b_o)
h_4 = o_4 * tanh(C_4)

print(f"h_4 = {h_4:.4f}")

# Final prediction using linear transformation
y_pred = np.dot(W_y, [h_4, h_4, h_4, h_4]) + b_y
print(f"\nPredicted next value: {y_pred:.4f}")
print("Expected: ~3.8 ")