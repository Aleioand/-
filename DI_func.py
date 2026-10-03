import numpy as np
import torch

def GaussianMatrix(X,sigma):
    G = torch.mm(X, X.T)
    K = 2*G-(torch.diag(G).reshape([1,G.size()[0]]))
    K = 1/(2*sigma**2)*(K-(torch.diag(G).reshape([G.size()[0],1])))
    K = torch.exp(K)
    
    return K

def Directed_Information(variable1,variable2,variable3,alpha = 1.01):
    sigma = (variable1.size(0))**(-1/(4+(variable1.size(1))))
    input1 = variable1
    K_x = GaussianMatrix(input1,sigma)/(input1.size(0))
    L_x = torch.linalg.eigvalsh(K_x)
    lambda_x = torch.abs(L_x)
    
    #lambda_x = L_x
    #H_x = (1/(1-alpha))*torch.log2((torch.sum(lambda_x ** alpha)))
    
    
    
    input2 = variable2
    K_y = GaussianMatrix(input2,sigma)/(input2.size(0))
    L_y = torch.linalg.eigvalsh(K_y)
    lambda_y = torch.abs(L_y)
    #lambda_y = L_y
    H_y = (1/(1-alpha))*torch.log2((torch.sum(lambda_y ** alpha)))
    
    
    
    input3 = variable3
    K_z = GaussianMatrix(input3,sigma)/(input3.size(0))
    L_z = torch.linalg.eigvalsh(K_z)
    lambda_z = torch.abs(L_z)
    #lambda_y = L_y
    #H_z = (1/(1-alpha))*torch.log2((torch.sum(lambda_z ** alpha)))
    
    
    K_xz = K_x*K_z*(input1.size(0))
    K_xz = K_xz / torch.sum(torch.diag(K_xz))
    
    L_xz = torch.linalg.eigvalsh(K_xz)
    lambda_xz = torch.abs(L_xz)
    #lambda_xy = L_xy
    H_xz =  (1/(1-alpha))*torch.log2((torch.sum(lambda_xz ** alpha)))
    
    
    K_xyz = K_x*K_y*K_z*(input1.size(0))
    K_xyz = K_xyz / torch.sum(torch.diag(K_xyz))
    
    L_xyz = torch.linalg.eigvalsh(K_xyz)
    lambda_xyz = torch.abs(L_xyz)
    #lambda_xy = L_xy
    H_xyz =  (1/ (1 - alpha))* torch.log2((torch.sum(lambda_xyz ** alpha)))
    
    #mutual_information = H_x + H_y - H_xy
    return H_y - (H_xyz - H_xz)

def build_DI_inputs(cm, r, L):

    N = len(r)
    M = N - L
    X = np.zeros((M, L),  np.float32)
    Y = np.zeros((M, 1),  np.float32)
    Z = np.zeros((M, L),  np.float32)
    for i in range(M):
        X[i, :] = cm[i : i+L]
        Y [i, 0] = r[i+L]
        Z[i, :] = r[i : i+L]
    return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Z)
