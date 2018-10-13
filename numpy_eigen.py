from scipy.linalg import eigh as eig
import pickle
import numpy as np

with open('matrices.pkl', 'rb') as f:
    matrices = pickle.load(f)
W = matrices['W']
y_prob = matrices['y_prob']

D = np.diag(y_prob[0])
A = (D - np.matmul(y_prob.T, y_prob))

sigma_A, U_A = eig(A)
sigma_A_sqrt = np.sqrt(sigma_A)
sigma_A_sqrt = np.diag(sigma_A_sqrt)
B = np.matmul(W, U_A)
B = np.matmul(B, sigma_A_sqrt)

BTB = np.matmul(B.T, B)
sigma_B_sq, V_B = eig(BTB)
print('sigma_B_sq', sigma_B_sq.tolist())

sigma_B_inv = np.reciprocal(np.sqrt(sigma_B_sq))
sigma_B_inv = np.diag(sigma_B_inv)

HEV = np.matmul(V_B, sigma_B_inv)
HEV = np.matmul(B, HEV)

print(HEV.shape)
