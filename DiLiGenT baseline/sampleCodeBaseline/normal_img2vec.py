import numpy as np

def normal_img2vec(N_est, m) : # N_est : (512, 612, 3)

    p = len(m)
    
    Nx = np.reshape(np.transpose(N_est[:,:,0]), -1)
    Ny = np.reshape(np.transpose(N_est[:,:,1]), -1)
    Nz = np.reshape(np.transpose(N_est[:,:,2]), -1)

    S = np.zeros((p, 3))

    for i in range(len(m)):
        S[i, 0] = Nx[m[i]]
        S[i, 1] = Ny[m[i]]
        S[i, 2] = Nz[m[i]]

    return S 