import numpy as np
import torch
import scipy.io as spo
def DmaxD(x, p, distf):
    I = np.array([], dtype=np.int32)
    if distf == 'E':
        distf = lambda x,y: np.matmul((x - y).transpose(), x-y)
    elif distf == 'L1':
        distf = lambda x,y: np.sum(np.abs(x - y))
    elif distf == 'Gaussian':
        distf = lambda x,y: 100*np.sum(np.exp(-np.square(x - y)/2))
    D, N = x.shape
    d = np.zeros([p, N])
    for i in range(N):
        d[0,i] = distf(x[:,i], np.zeros(D))
    I = np.append(I,np.argmax(d[0,:]))

    for i in range(N):
        d[0,i] = distf(x[:,i], x[:,I[0]])

    for v in range(1,p):
        row_1 = np.concatenate([d[:v,I[:v]], np.ones([v,1])], axis=1)
        row_2 = np.append(np.ones([1,v]),0).reshape(1,-1)
        D = np.concatenate([row_1, row_2])
        D = np.linalg.inv(D)
        V = np.zeros(N)
        for i in range(N):
            xx = np.append(d[:v,i],1).reshape(-1,1)
            tmp = np.matmul(xx.transpose(1,0),D)
            V[i] = np.matmul(tmp,xx)
        I = np.append(I, np.argmax(V))
        for i in range(N):
            d[v,i]=distf(x[:,i], x[:,I[v]])
    I = np.sort(I)

    return I


if __name__ == '__main__':
    a = spo.loadmat('HSI_data/CupriteS/CupriteS1_R188.mat')
    col = a['nCol']
    row = a['nRow']
    img = a['Y']
    band = img.shape[0]
    img = img / img.max()
    I = DmaxD(img, 12, 'L1')
    print(1)