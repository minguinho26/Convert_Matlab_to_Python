import numpy as np
import scipy.io
import cv2

import util as ut


def main_Baseline(dataset_path) : 
    dataFormat = 'PNG'
    dataNameStack = np.array(['ball', 'cat', 'pot1', 'bear', 'pot2', 'buddha', 'goblet', 'reading', 'cow', 'harvest'])

    for testId in range(10) :
        dataName = dataNameStack[testId] +  dataFormat
        datadir = dataset_path + '/pmsData/' + dataName
        bitdepth = 16
        gamma = 1
        resize = 1
        data = ut.load_datadir_re(datadir, bitdepth, resize, gamma) 

        mask1 = 0
        if data['mask'].shape[2] == 1 :
            mask1 = data['mask']/255.0
        else : 
            mask1 = np.add(np.add(0.299*data['mask'][:,:,0], 0.587*data['mask'][:,:,1]), 0.114*data['mask'][:,:,2])/255.0 # (512, 612). 차이 없음
        
        mask3 = np.transpose(np.array([np.transpose(mask1), np.transpose(mask1), np.transpose(mask1)])) # 차이 없음

        m = np.reshape(np.array(np.argwhere(np.reshape(np.transpose(mask1), -1) == 1), dtype = object), -1) # 값을 살릴 위치만 나온다

        Normal_L2 = ut.L2_PMS(data, m)

        _Normal = np.transpose(Normal_L2)
        _mask3 = np.transpose(mask3)

        write_img = []

        for i in range(3) : 
            write_img.append(np.multiply( ((_Normal[i, :, :] + 1) * 128).astype(np.uint8), _mask3[i, :, :].astype(np.uint8)))

        write_img = np.transpose(write_img)

        cv2.imwrite(dataName + '_Normal_l2.png', write_img)
        Normal_L2_mat={'Normal_L2_test':Normal_L2}
        scipy.io.savemat(dataName + '_Normal_l2.mat', Normal_L2_mat)

        print(dataName + " created")