import torch
import torch.nn as nn
import numpy as np
import random
from torch.nn import init

def corr_map(input, dist):
    row = input.shape[0]
    col = input.shape[1]
    bands = input.shape[2]
    win_radius = 1
    new_map = np.zeros((row,col))
    if dist == 'cos':
        dist = lambda x,y: ((x*y.reshape(1,-1)).sum(1)/(np.linalg.norm(x,2,axis=1)*np.linalg.norm(y,2))).mean()
    if dist == 'Gaussian':
        dist = lambda x,y: np.exp(-np.square(x - y.reshape(1,-1))/0.005).mean()
    for i in range(row):
        for j in range(col):
            start_row = np.maximum(0,i - win_radius)
            start_col = np.maximum(0,j - win_radius)
            end_row = np.minimum(row, i + win_radius + 1)
            end_col = np.minimum(col, j + win_radius + 1)
            row_index = [i_ for i_ in range(start_row, end_row)]
            col_index = [i_ for i_ in range(start_col, end_col)]
            index = [[i_,j_] for i_ in row_index for j_ in col_index]
            index_tmp = index.copy()
            for k in range(len(index)):
                if index[k] == [i,j]:
                    index_tmp.pop(k)
            index = index_tmp
            index = np.array(index)
            batch = input[index[:,0], index[:,1]].reshape(-1,bands)
            center_spectrum = input[i,j]
            new_map[i,j] = dist(batch, center_spectrum)
    return new_map


class BatchNorm1d(nn.BatchNorm1d):
    # pass
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
class my_Linear(nn.Module):
    def __init__(self, init):
        super(my_Linear, self).__init__()
        self.weight = torch.Tensor(init).cuda()
        self.weight = nn.Parameter(self.weight)

    def forward(self, x):
        output = torch.matmul(x, self.weight)
        return output

class mynet_decoder(nn.Module):
    def __init__(self, init):
        super(mynet_decoder, self).__init__()
        self.layer = my_Linear(init)
    def forward(self, x):
        x = self.layer(x)
        return x


class mynet_encoder(nn.Module):
    def __init__(self, input, total_num, ini):
        super(mynet_encoder, self).__init__()
        self.ini_1 = ini
        self.mask = nn.ModuleList()
        self.ini_2 = np.linalg.pinv(np.matmul(ini.transpose(), ini))
        self.encoder = nn.ModuleList()
        self.rho = nn.Parameter(torch.ones(1,total_num).cuda()*2)
        # self.rho = nn.Parameter(torch.ones(1,total_num)*2)
        self.mask += [
            nn.Linear(input, 32, bias=False),
            nn.BatchNorm1d(32, affine=False),
            nn.ELU(inplace=True),
            nn.Linear(32, total_num, bias=False),
            nn.BatchNorm1d(total_num, affine=False),
        ]

        # self.encoder += [
        #     nn.Linear(input, 64, bias=True),
        #     # nn.BatchNorm1d(64, affine=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 32, bias=True),
        #     # nn.BatchNorm1d(32, affine=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, total_num, bias=False),
        #     nn.BatchNorm1d(total_num, affine=True),
        # ]
        self.encoder += [
            my_Linear(self.ini_1),
            nn.ReLU(inplace=True),
            my_Linear(self.ini_2),
            BatchNorm1d(total_num, affine=True),

        ]

    def forward(self, x):
        mask = x
        for layer in self.mask:
            mask = layer(mask)
        mask = mask * torch.max(self.rho, self.rho.new_full(self.rho.shape, 1.0))
        mask = nn.Sigmoid()(mask)
        # x = x
        for layer in self.encoder:
            x = layer(x)
        x = x * torch.max(self.rho, self.rho.new_full(self.rho.shape, 1.0))
        # x = nn.ReLU(inplace=True)(x)

        x_ = nn.Softmax(dim=1)(x)

        x = x_*mask
        # x = x_
        return x,  x_

def loss(output, target, x_,ab_pre, weight,  total_num):
    mse = torch.mean((torch.abs(output - target + 1e-8))**0.7)
    kl =  -1*(torch.log((output * target).sum(dim=1))
          - torch.log(torch.norm(output, 2, 1))
          - torch.log(torch.norm(target, 2, 1))).mean()
    orth = torch.matmul(ab_pre.transpose(1,0), ab_pre)/torch.matmul(ab_pre.norm(2,dim=0).view(-1,1),ab_pre.norm(2,dim=0).view(1,-1))
    loss = 1*mse +\
           1*kl-\
           10 * ((weight < 0).float() * weight).mean() +\
           0.5 * orth.abs().mean() + \
           0.1 * ab_pre[:, :total_num].abs().mean() #+\
           # 0.1 * x_[:, :total_num].abs().mean() #+\
           #0. * ((ab_pre[:, :total_num].abs().mean(dim=1))**(1)).mean()

    return loss
