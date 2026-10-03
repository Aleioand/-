import numpy as np
import torch
from DI_func import Directed_Information, build_DI_inputs

def generate_c_series(M, N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    c = np.zeros((M, N), dtype=float)
    for n in range(N-1):
        c[:, n+1] = (c[:, n] + np.random.randn(M)) > 0
    return c.astype(float)


def generate_R(c, alpha=0.1, noise_scale=0.01, seed = None):
    
    M, N = c.shape
    if seed is not None:
        np.random.seed(seed)
    R = np.zeros(N, dtype=float)
    R[:2] = noise_scale * np.random.randn(2)
    for n in range(1, N-1):

        term1 = alpha * R[n]
        term2 = (c[0,n] != c[0, n-1])           
        term3 = (c[1,n] != c[1, n-1]) ** c[2,n]  
        noise = noise_scale * np.random.randn()
        R[n+1] = term1 + term2 + term3 + noise
    R = (R - np.min(R)) / (np.max(R) - np.min(R))
    return R


if __name__ == "__main__":
    M = 6      
    N = 1024   
    L = 2

    
    c = generate_c_series(M, N)

    
    R = generate_R(c)

    
    print("c[0] head:", c[0,:10])
    print("c[1] head:", c[1,:10])
    print("c[2] head:", c[2,:10])
    print("R[0..10]:", np.round(R[:11], 3))


    dis = []
    for m in range(M):
        cm = c[m]
        X, Y, Z = build_DI_inputs(cm, R, L)
        dis.append(Directed_Information(X, Y, Z , 1.01))

# Display results
    dis = np.array(dis)
    top3 = dis.argsort()[::-1][:3]
    print("DI scores:", dis)
    print("Top-3 series indices (ground truth 0,1,2):", top3)

        
