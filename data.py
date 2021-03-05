from scipy.io import savemat
import numpy as np
import scipy.io as spo


def read_data(data_name):
    member = abundance = member_name = None
    if data_name == 'CupriteS':
        a = spo.loadmat('./HSI_data/CupriteS/CupriteS1_R188.mat')
        row = np.array(int(a['nCol']))
        col = np.array(int(a['nRow']))
        img = a['Y']
        band = img.shape[0]
        select_ind = a['SlectBands'].squeeze()
        img[img>60000] = 0
        img = img / 10000
        img[img<0]=0
        total_num = 12
        endmember = spo.loadmat('./HSI_data/CupriteS/groundTruth_Cuprite_nEnd12.mat')
        endmember1 = endmember['M']
        member = endmember1[select_ind]
        file = 'CupriteS'
    elif data_name == 'WashingtonDC':
        a = spo.loadmat('../HSI_data/washingtonDC/washingtonDC.mat')
        img = a['dc'].astype(np.float32)[:256,:256]
        col = img.shape[1]
        row = img.shape[0]
        band = img.shape[2]
        img = img / img.max()
        img[img<0] = 0
        img = img.reshape(-1,band)
        img = img.transpose(1,0)
        total_num = 8
        file = 'WashingtonDC'
    elif data_name == 'Urban':
        a = spo.loadmat('./HSI_data/Urban/Urban_R162.mat')
        col = a['nCol'].squeeze()
        row = a['nRow'].squeeze()
        img = a['Y']
        band = img.shape[0]
        img = img/img.max()
        endmember = spo.loadmat('./HSI_data/Urban/end5_groundTruth.mat')
        member = endmember['M']
        abundance = endmember['A']
        file = 'Urban'
        total_num = 6
        member_name = endmember['cood']
    elif data_name == 'JasperRidge':
        a = spo.loadmat('./HSI_data/JasperRidge/jasperRidge2_R198.mat')
        col = a['nCol'].squeeze()
        row = a['nRow'].squeeze()
        img = a['Y']
        band = img.shape[0]
        img = img/img.max()
        endmember = spo.loadmat('./HSI_data/JasperRidge/end4.mat')
        member = endmember['M']
        abundance = endmember['A']
        total_num = 4
        member_name = endmember['cood']
        file = 'JasperRidge'
    elif data_name == 'gulfport':
        a = spo.loadmat('HSI_data/gulfport/gulfport_data.mat')
        img = a['Y']
        row = np.array(img.shape[0])
        col = np.array(img.shape[1])
        band = img.shape[2]
        img = img.reshape(-1,band)
        img[img<0]=0
        img = img.transpose(1,0)
        total_num = 6
        file = 'gulfport'
    elif data_name == 'pavia':
        a = spo.loadmat('../HSI_data/pavia/PaviaU.mat')
        img = a['paviaU']
        row = img.shape[0]
        col = img.shape[1]
        band = img.shape[2]
        img = img.reshape(-1,band)
        img[img<0]=0
        img=img/img.max()
        img = img.transpose(1,0)
        total_num = 6
        file = 'pavia'
    elif data_name == 'Samson':
        a = spo.loadmat('./HSI_data/Samson/samson.mat')
        col = a['nCol'].squeeze()
        row = a['nRow'].squeeze()
        band = a['nBand'].squeeze()
        img = a['V']
        endmember = spo.loadmat('./HSI_data/Samson/end3.mat')
        member = endmember['M']
        abundance = endmember['A']
        total_num = 3
        member_name = endmember['cood']
        file = 'Samson'
    elif data_name == 'sandiego':
        a = spo.loadmat('./HSI_data/Sandiego/sandiego.mat')
        img = a['ans']
        img[img<0]=0
        col = np.array(img.shape[1])
        row = np.array(img.shape[0])
        band = img.shape[2]
        img = img.reshape(-1, band).transpose(1,0)
        total_num = 8
        file = 'sandiego'
    elif data_name == 'TK':
        a = spo.loadmat('../HSI_data/TK/TK1.mat')
        # print(1)
        img = a['img']
        img[img<0]=0
        img=img/img.max()
        col = np.array(img.shape[1])
        row = np.array(img.shape[0])
        band = img.shape[2]
        img = img.reshape(-1, band).transpose(1,0)
        total_num = 6
        file = 'TK'
    else:
        assert 1
    return img, col, row, band, total_num, member, abundance, file, member_name