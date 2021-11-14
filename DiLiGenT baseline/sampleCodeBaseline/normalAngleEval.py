import numpy as np

def normalAngleEval(N_true, N_est, maskindex) :

    masklength = len(maskindex);
    N_true1  = np.zeros(masklength, 3);
    N_est1  = np.zeros(masklength, 3);
    height, width, color = N_true.shape

    for i in range(3) :
        temp1 = N_true[:, :, i]
        N_true1[:,i] = temp1[maskindex]

        temp2 = N_est[:, :, i]
        N_est1[:,i] = temp2[maskindex]

    dotM = np.zeros(masklength)

    for i in range(masklength) :
        dotM[i] = np.dot(N_true1[:,i], np.transpose(N_est1[:,i]))

    angle = np.divide(np.multiply(180, np.arccos(dotM)), np.pi)

    angle = angle.astype(np.float64)

    meanA = np.mean(angle)
    stdA = np.std(angle)
    medianA = np.median(angle)
    maxA = np.max(angle)

    angleImg = np.zeros(height*width)
    for i in range(masklength) :
        angleImg[maskindex[i]] = angle[i]
    
    angleImg = np.reshape(angleImg, (height, width))

    return meanA, medianA, stdA, maxA, angleImg, angle