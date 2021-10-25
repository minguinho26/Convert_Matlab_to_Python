import numpy as np
import scipy.io
import cv2

import util as ut


# 각 이미지별로 Surface Normal을 출력해주는 함수
def main_Baseline_per_Image_folder(Image_Folder_path) : 
    
    path_split = Image_Folder_path.split("/")

    objaName = path_split[-1]
    bitdepth = 16
    gamma = 1
    resize = 1
    data = ut.load_datadir_re(Image_Folder_path, bitdepth, resize, gamma) 

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

    cv2.imwrite(objaName + '_Normal_l2.png', write_img)
    Normal_L2_mat={'Normal_L2_test':Normal_L2}
    scipy.io.savemat(objaName + '_Normal_l2.mat', Normal_L2_mat)

    print(objaName + " created")