import numpy as np
from scipy import io
import cv2
import matplotlib.pyplot as plt

from normal_img2vec import *

# dataset_path : DiLiGenT dataset의 경로를 입력
# dataset_path : root path of DiLiGenT dataset
def mainNonLembert_boxPlot_perMethod(dataset_path):
    dataNameStack = np.array(['ball', 'cat', 'pot1', 'bear', 'pot2', 'buddha', 'goblet', 'reading', 'cow', 'harvest'])
    numData = len(dataNameStack)

    methodStack = np.array(['L2', 'ACCV10Wu', 'CVPR12Ikehata', 'ICCV05Goldman', 'CVPR08Alldrin', 'CVPR10Higo', 'ECCV12Shi', 'CVPR12Shi', 'CVPR14Ikehata'])
    numMethod = len(methodStack)

    methodDispName = np.array(['BASELINE', 'WG10', 'IW12', 'GC10', 'AZ08', 'HM10', 'ST12', 'ST14', 'IA14'])

    dataFormat = 'PNG'
    dataDir = dataset_path + '/pmsData'
    resultDir = dataset_path + '/estNormalNonLambert'

    numPx = [15791,45200,57721,41512,35278,44864,26193,27654,26421,57342]
    pMax = max(numPx);

    angErrStack = []

    for iM in range(numMethod) :
        print('Working on Method no.:',  str(iM + 1), " (", methodStack[iM], ")")
        angErrMat = np.empty((pMax,numData))
        angErrMat[:] = np.nan
        for iD in range(numData) :
            # 여기서 normal을 불러온다. 즉, 여기서 mat파일을 불러온다.
            # PMS방식별로 측정한 normal, 실제 normal, 마스크를 불러온다
            mat_file_path =  resultDir + '/' + dataNameStack[iD] + dataFormat + '_Normal_' + methodStack[iM] + '.mat'
            Normal_est_mat_file = io.loadmat(mat_file_path)
            Normal_est = 0
            if iM == 0:
                Normal_est = Normal_est_mat_file['Normal_L2']
            else:
                Normal_est = Normal_est_mat_file['Normal_est']
            
            mat_file_path = dataDir + '/' + dataNameStack[iD] + dataFormat + '/' + 'Normal_gt.mat'
            Normal_gt_mat_file = io.loadmat(mat_file_path)
            Normal_gt = Normal_gt_mat_file['Normal_gt']


            mask_path = dataDir + '/' + dataNameStack[iD] + dataFormat + '/' + 'mask.png'
            img = cv2.imread(mask_path).astype(np.float64)
            mask = np.array(np.transpose( [np.transpose(img[:, :, 2]), np.transpose(img[:, :, 1]), np.transpose(img[:, :, 0]) ]))
            mask = np.add(np.add(0.299*mask[:,:,0], 0.587*mask[:,:,1]), 0.114*mask[:,:,2])
            th, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = np.array(mask)/255.0

            m = 0
            N_gt = 0
    
            m = np.reshape(np.array(np.argwhere(np.reshape(np.transpose(mask), -1) == 1), dtype = object), -1) # 값을 살릴 위치만 나온다
            N_gt = normal_img2vec(Normal_gt, m)
            N_est = normal_img2vec(Normal_est, m).astype(np.float64)

            dot_value = np.multiply(N_gt[:, 0], N_est[:, 0]) + np.multiply(N_gt[:, 1], N_est[:, 1]) + np.multiply(N_gt[:, 2], N_est[:, 2]) # 오차 X

            angErr = []
            for i in range(len(dot_value)) :
                # 범위 내에 있는 값은 잘나온다.
                if abs(dot_value[i]) > 1.0 :
                    temp = 0
                    angErr.append(temp)
                else :
                    temp = np.real((np.arccos(dot_value[i]).astype(np.float64))) * (180.0 / np.pi)
                    angErr.append(temp)

            angErr = np.array(angErr)
            print('MeanErr-' + methodStack[iM] + '/' + dataNameStack[iD] + ":" + str(np.mean(angErr)))

            angErrMat[0:len(m), iD] = angErr;
        angErrStack.append(angErrMat)

    fs = 8  # fontsize
    hFig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 20), sharey=True)
    for i in range(numMethod) :
        y = i//3
        x = i%3

        numPx = [15791,45200,57721,41512,35278,44864,26193,27654,26421,57342]
        print(angErrStack[i].shape)
        data = [angErrStack[i][:,0][:numPx[0]], angErrStack[i][:,1][:numPx[1]], angErrStack[i][:,2][:numPx[2]], angErrStack[i][:,3][:numPx[3]], angErrStack[i][:,4][:numPx[4]], angErrStack[i][:,5][:numPx[5]], angErrStack[i][:,6][:numPx[6]], angErrStack[i][:,7][:numPx[7]], angErrStack[i][:,8][:numPx[8]], angErrStack[i][:,9][:numPx[9]]] 
        axes[y, x].boxplot(data, labels=[0,1,2,3,4,5,6,7,8,9], showfliers=False)
        axes[y, x].set_title(methodDispName[i], fontsize=fs)
        axes[y, x].set_ylim([0, 36])
       
        # Calculat mean by excluding NaN
        mx = np.zeros((1, numData))
        
        for j in range(numData) :
            mx[0,j] = np.mean(angErrStack[i][:numPx[j],j])
        axes[y, x].scatter(np.array([1,2,3,4,5,6,7,8,9,10]), mx[0,:])
