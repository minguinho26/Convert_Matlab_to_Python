import numpy as np

# Re-organize from vector to image 
def normal_vec2img(N_est, height, width, m) :
    
    p = len(m)

    n_x = np.zeros(height * width, 1)
    n_y = np.zeros(height * width, 1)
    n_z = np.zeros(height * width, 1)


    N_est_x =  np.reshape(np.transpose(N_est[:,:,0]), -1)
    N_est_y =  np.reshape(np.transpose(N_est[:,:,1]), -1)
    N_est_z =  np.reshape(np.transpose(N_est[:,:,2]), -1)

    for i in range(p) :
        n_x[m[i]] = N_est_x[i]
        n_y[m[i]] = N_est_y[i]
        n_z[m[i]] = N_est_z[i]

    n_x = np.reshape(n_x, (height, width))
    n_y = np.reshape(n_y, (height, width))
    n_z = np.reshape(n_z, (height, width))

    N = np.zeros(height, width, 3)

    N[:, :, 0] = n_x;
    N[:, :, 1] = n_y;
    N[:, :, 2] = n_z;

    N[np.isnan(N)] = 0;

    N_est_img = N;

    return N_est_img
