import numpy as np
import scipy.io
import cv2
import re
import matlab.engine

def L2_PMS(data, m) :
    m = np.reshape(m, -1).astype(np.int32)
    light_dir = np.transpose(data['s']) # 매트랩과 같다.
    f = light_dir.shape[1]

    height, width, color = data['mask'].shape
    p = len(m)

    I = np.zeros((p, f)) # (15791, 96)

    for i in range(f) : # frame
        img = data['imgs'][i]
        # 0.299 * R + 0.587 * G + 0.114
        img = np.add(np.add(0.299*img[:,:,0], 0.587*img[:,:,1]), 0.114*img[:,:,2]).astype(np.float64) # 여기서 512*612*3 -> 512*612
        # gray scale을 구하는 과정에서 차이가 발생 -> 오차들이 하나둘씩 모여 I에서 큰 차이 발생
        I[:, i] = (np.reshape(np.transpose(img), -1)[m]).copy()
    
    L = light_dir.copy()
    S_hat = np.dot(I, np.linalg.pinv(L))

    S = np.zeros(S_hat.shape)

    signX = 1
    signY = 1
    signZ = 1

    for i in range(p) :
        length = np.round(np.sqrt( S_hat[i, 0]*S_hat[i, 0] + S_hat[i, 1]*S_hat[i, 1] + S_hat[i, 2]*S_hat[i, 2]), 4)
        S[i, 0] = np.multiply(np.divide(S_hat[i, 0] ,length), signX)
        S[i, 1] = np.multiply(np.divide(S_hat[i, 1] ,length), signY)
        S[i, 2] = np.multiply(np.divide(S_hat[i, 2] ,length), signZ)

    n_x = np.resize(np.zeros(1), (height*width, 1)) # (313344,1)
    n_y = np.resize(np.zeros(1), (height*width, 1))
    n_z = np.resize(np.zeros(1), (height*width, 1))

    # 법선 벡터. 여기 통과
    for i in range(p):
        n_x[m[i]] = S[i, 0].copy()
        n_y[m[i]] = S[i, 1].copy()
        n_z[m[i]] = S[i, 2].copy()

    n_x = np.reshape(n_x, -1)
    n_y = np.reshape(n_y, -1)
    n_z = np.reshape(n_z, -1)

    _N = np.zeros((height*width, 3))

    _N[: ,0] = np.transpose(n_x.copy())
    _N[:, 1] = np.transpose(n_y.copy())
    _N[:, 2] = np.transpose(n_z.copy())

    # Transpose를 하니까 (B, G, R)순으로, 다시 말해 법선 벡터의 (Z, Y, X)순으로 넣어준다.
    N = np.transpose(np.array([np.reshape(_N[: ,2], (width, height)), np.reshape(_N[: ,1], (width, height)), np.reshape(_N[: ,0], (width, height))]))  

    N[np.isnan(N)] = 0

    return N

def imread_datadir_re(datadir, which_image, bitDepth, resize, gamma) :
    if not hasattr(imread_datadir_re, "zeros_mat"): imread_datadir_re.zeros_mat = 0 # it doesn't exist yet, so initialize it

    _E = cv2.imread(datadir['filenames'][which_image], cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR).astype(np.float64)
    E = np.array(np.transpose( [np.transpose(_E[:, :, 2]), np.transpose(_E[:, :, 1]), np.transpose(_E[:, :, 0]) ])) # (512, 612, 3)

    # E[:, :, 0] = np.power(E[:, :, 0]/((2**bitDepth)-1), float(gamma))
    # E[:, :, 1] = np.power(E[:, :, 1]/((2**bitDepth)-1), float(gamma))
    # E[:, :, 2] = np.power(E[:, :, 2]/((2**bitDepth)-1), float(gamma))  

    # np.power 빼니까 연산속도 향상
    E[:, :, 0] = E[:, :, 0]/((2**bitDepth)-1)
    E[:, :, 1] = E[:, :, 1]/((2**bitDepth)-1)
    E[:, :, 2] = E[:, :, 2]/((2**bitDepth)-1) 

    E = np.transpose(cv2.resize(np.transpose(E) , None , fx=resize ,fy=resize ,interpolation = cv2.INTER_NEAREST))

    H,W,C = E.shape

    E = np.reshape(E,(H*W, 3))

    if which_image == 0:
        imread_datadir_re.zeros_mat = np.zeros(E.shape).astype(np.float64)

    E[:,0] = np.multiply(E[:,0].astype(np.float64), 1.0/datadir['L'][which_image][0])
    E[:,1] = np.multiply(E[:,1].astype(np.float64), 1.0/datadir['L'][which_image][1])
    E[:,2] = np.multiply(E[:,2].astype(np.float64), 1.0/datadir['L'][which_image][2])

    E = np.maximum(E, imread_datadir_re.zeros_mat)

    E = np.reshape(E, (H,W,C))

    return E # (512, 612, 3)

def load_datadir_re(datadir, bitDepth, resize, gamma) :
    white_balance = np.array([[1,0,0], [0,1,0], [0,0,1]])
    
    # 파이썬 딕셔너리로 사용.
    data = {'s' : 0, 'L' : 0, 'filenames' : 0, 'mask' : 0, 'image' : 0}
    # 파일을 불러온다
    fid = open(datadir + '/light_directions.txt','r')

    light_direction = []

    fid_contents = fid.readlines()

    for i in range(len(fid_contents)) :
        num_in_line = (np.array((re.findall(r"[-+]?\d*\.*\d+", fid_contents[i])))).astype(np.float32)
        light_direction.append(num_in_line) # 빛의 방향이 [x, y, z]

    data['s'] = np.array(light_direction)

    fid.close()

    light_intensity = []
    fid = open(datadir + '/light_intensities.txt','r')

    fid_contents = fid.readlines()
    for i in range(len(fid_contents)) :
        num_in_line = np.array((re.findall(r"[-+]?\d*\.*\d+", fid_contents[i]))).astype(np.float32)
        light_intensity.append(num_in_line)

    data['L'] = np.dot(np.array(light_intensity), white_balance)

    fid.close()

    fid = open(datadir + '/filenames.txt','r')
    data['filenames'] = fid.readlines()

    for i in range(len(data['filenames'])) :
        data['filenames'][i] = datadir + '/' + data['filenames'][i][:-1]

    fid.close()

    mask_path = datadir + '/mask.png'
    _mask = cv2.imread(mask_path).astype(np.float64)
    data['mask'] = np.array(np.transpose( [np.transpose(_mask[:, :, 2]), np.transpose(_mask[:, :, 1]), np.transpose(_mask[:, :, 0]) ])).astype(np.float32)
    data['mask']= cv2.resize(data['mask'] ,None , fx=resize ,fy=resize ,interpolation = cv2.INTER_NEAREST)


    imgs = [] # 96*1사이즈 : 이미지 불러오는거

    # data_ = copy.deepcopy(data)
    
    # 여기서 오래 걸림
    for i in range(len(data['filenames'])) :
        np_maximum = imread_datadir_re(data, i, bitDepth, resize, gamma)
        imgs.append(np_maximum)

    data['imgs'] = np.array(imgs) # (96, 512, 612, 3)

    return data