import numpy as np
import math
from scipy import io
import cv2
import matplotlib.pyplot as plt

from normal_img2vec import *

# dataset_path : DiLiGenT dataset의 경로를 입력
# dataset_path : root path of DiLiGenT dataset
def mainUncalib_boxPlot_perMethod(dataset_path):
    dataNameStack = np.array(['ball', 'cat', 'pot1', 'bear', 'pot2', 'buddha', 'goblet', 'reading', 'cow', 'harvest'])
    numData = len(dataNameStack)

    # Evaluated methods directory
    methodStackDir = np.array(['CVPR12FavaroCombined', 'CVPR12FavaroCombined', 'CVPR12FavaroCombined', 'CVPR13Wu', 'optimalGBR' 'optimalGBR'])
    # Evaluated methods name
    methodStack = np.array(['CVPR07Alldrin', 'CVPR10Shi', 'CVPR12Favaro', 'CVPR13Wu', 'optA', 'optG'])
    numMethod = len(methodStack)

    # Display name for methods
    methodDispName = np.array(['AM07', 'SM10', 'PF14', 'WT13', 'Opt. A', 'Opt. G'])
    # Put LM13 last, since it requires special downsample
    # The sub-figure (box plots) sequence in the paper is [1, 2, 3, 4, 7, 5, 6]

    dataFormat = 'PNG'
    dataDir = dataset_path + '/pmsData'
    resultDir = dataset_path + '/estNormalUncalib'

    availableData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    availableMethod = [1, 2, 3, 4, 5, 6, 7]

    # Number of normals for each data
    numPx = [15791,45200,57721,41512,35278,44864,26193,27654,26421,57342]; # 오브젝트별 법선 벡터 개수
    pMax = max(numPx);
    # For better visualization
    # cutVal = 60;

    angErrStack = [] # cell : 모든 데이터를 저장할 수 잇는 데이터형

    mxAll = np.zeros((numMethod, numData)) # (6, 10). 평균 오차 저장

    for iM in range(0, numMethod) :
        print('Working on Method no.:',  str(iM + 1), " (", methodStack[iM], ")")
        angErrMat = np.zeros((pMax, numData))
        angErrMat[:, :] = np.nan
        for iD in range(0, numData) :

            # Load estimated results 'Normal_est'
            mat_file_path =  resultDir + '/' + dataNameStack[iD] + dataFormat + '_Normal_' + methodStack[iM] + '.mat'
            Normal_est_mat_file = io.loadmat(mat_file_path) # 딕셔너리로 불러온다. 그런데 사이즈가 (512, 612, 3)이다. 여기서 키가 'Normal_est'인 값들이 우리가 원하는 값이다.
            Normal_est = Normal_est_mat_file['Normal_est'] # 3개의 채널로 구성된 배열

            # Load ground truth normal 'Normal_gt'
            mat_file_path = dataDir + '/' + dataNameStack[iD] + dataFormat + '/' + 'Normal_gt.mat'
            Normal_gt_mat_file = io.loadmat(mat_file_path)
            Normal_gt = Normal_gt_mat_file['Normal_gt'] # Normal_gt에 해당하는 값만 불러오자

            # Load masks(이미지를 넘파이 배열로 불러왔다)
            mask_path = dataDir + '/' + dataNameStack[iD] + dataFormat + '/' + 'mask.png'
            # print(mask_path)
            img = cv2.imread(mask_path).astype(np.float64)
            mask = np.array(np.transpose( [np.transpose(img[:, :, 2]), np.transpose(img[:, :, 1]), np.transpose(img[:, :, 0]) ]))
            mask = np.add(np.add(0.299*mask[:,:,0], 0.587*mask[:,:,1]), 0.114*mask[:,:,2])
            th, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = np.array(mask)/255.0

            resize = 0.0

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
            mxAll[iM, iD] = np.mean(angErr)
            angErrMat[0:len(angErr), iD] = np.transpose(angErr).copy()
        angErrStack.append(angErrMat) # (6, 57221, 10) 형식으로 반환

    angErrStack_mat={'angErrStack_test':angErrStack}
    io.savemat(dataset_path + '/' +'_angErrStack_test.mat', angErrStack_mat)

    numPx = [15791,45200,57721,41512,35278,44864,26193,27654,26421,57342]
    methodDispName = np.array(['AM07', 'SM10', 'PF14', 'WT13', 'Opt. A', 'Opt. G'])

    fs = 8  # fontsize

    # demonstrate how to toggle the display of different elements:
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 12), sharey=True)

    for i in range(6) :
        data = [angErrStack[i][:,0][:numPx[0]], angErrStack[i][:,1][:numPx[1]], angErrStack[i][:,2][:numPx[2]], angErrStack[i][:,3][:numPx[3]], angErrStack[i][:,4][:numPx[4]], angErrStack[i][:,5][:numPx[5]], angErrStack[i][:,6][:numPx[6]], angErrStack[i][:,7][:numPx[7]], angErrStack[i][:,8][:numPx[8]], angErrStack[i][:,9][:numPx[9]]] 
        y = i//3
        x = i%3
        axes[y, x].boxplot(data, labels=[0,1,2,3,4,5,6,7,8,9], showfliers=False)
        axes[y, x].set_title(methodDispName[i], fontsize=fs)

    plt.show()