from scipy.io import savemat
from net_factory import *
import matplotlib.pyplot as plt
import torch.optim as optim
from DmaxD import *
# from config_linear import args
from data import read_data
from utils import show_unmix_result


batch = 1024
lr = 0.001
epoch = 40
mix_up_threshold = 0.8
repeat_time = 10
data_name = 'Samson'
# CupriteS, WashingtonDC, Urban, JasperRidge, gulfport, pavia, Samson, sandiego, TK
# setup_seed(110)


img, col, row, band, total_num, endmember, abundance,file, endmember_name = read_data(data_name)
endmember = torch.Tensor(endmember).cuda()
abundance = torch.Tensor(abundance.transpose(1, 0)).cuda()
SAD_list = []
RMSE_list = []

im_ = img.transpose().reshape(row, col, -1)
mix_map = corr_map(im_, 'Gaussian')
clean_map = (mix_map >= mix_up_threshold).reshape(-1)

for times in range(repeat_time):
    input_data = torch.Tensor(img.copy().transpose(1,0)).cuda()
    data_index = np.arange(input_data.shape[0])
    np.random.shuffle(data_index)
    data_index = torch.Tensor(data_index).long().cuda()
    input_data = input_data[data_index]

    input_data = input_data.cpu().numpy()
    clean_map_shuffle = clean_map[data_index.cpu().numpy()]
    I = DmaxD(input_data[clean_map_shuffle].transpose(), total_num, 'L1')
    ini = input_data[clean_map_shuffle].transpose()[:,I]

    # plt.plot(ini)
    # plt.show()
    input_data = torch.Tensor(input_data).cuda()

    net_encoder = mynet_encoder(band, total_num, ini)
    net_decoder = mynet_decoder(ini.transpose())
    net_encoder = net_encoder.cuda()
    net_decoder = net_decoder.cuda()
    net_encoder.train()
    net_decoder.train()
    op_1 = optim.Adam(net_encoder.parameters(), lr=lr,
                          weight_decay=0.0005)
    op_2 = optim.Adam(net_decoder.parameters(), lr=lr,
                          weight_decay=0.0005)
    for ii in range(epoch):
        net_encoder.train()
        net_decoder.train()
        for i in range(0, input_data.shape[0], batch):
            end = np.minimum(i + batch, input_data.shape[0])
            input = input_data[i:end]
            ab, x_ = net_encoder(input)
            y = net_decoder(ab)
            losses = loss(y, input, x_, ab, net_decoder.layer.weight,
                           total_num)
            op_1.zero_grad()
            op_2.zero_grad()
            losses.backward()
            if ii%3 in [2]:
                op_2.step()
            else:
                op_1.step()
        if ii%10 == 0:
            print('epoch:', ii, losses.item())


    net_encoder.eval()
    net_decoder.eval()
    a, _ = net_encoder(torch.Tensor(img.transpose(1,0)).cuda())

    ind = a.std(dim=0)
    std, ind = ind.sort(descending=True)
    ind = ind[:total_num]
    a = a.view(int(row),int(col),total_num)
    a = a.permute(2,0,1).contiguous()[ind].permute(1,2,0)
    endmember_pre = net_decoder.layer.weight.transpose(1, 0)

    show_unmix_result(total_num, a, endmember_pre,show='False')

    new_end = []
    a = a.contiguous().view(-1,a.shape[-1])
    RMSE = lambda x, y: torch.sqrt(((x - y)**2).mean())
    SAD = lambda x, y: torch.acos(((x)*(y)).sum()/((x).norm(2)*(y).norm(2)))


    SAD_mat = torch.Tensor(a.shape[-1], abundance.shape[-1]).cuda()
    RMSE_mat = torch.Tensor(a.shape[-1], abundance.shape[-1]).cuda()

    for i in range(a.shape[-1]):
        for j in range(abundance.shape[-1]):
            RMSE_mat[i,j] = RMSE(a[:,i].contiguous(), abundance[:,j])
            SAD_mat[i,j] = SAD(endmember_pre[:,i], endmember[:,j])
    SAD, _ = SAD_mat.min(0)
    RMSE, _ = RMSE_mat.min(0)
    # print('SAD:',endmember_name, SAD.detach().cpu().numpy())
    # print('RMSE:', endmember_name, RMSE.detach().cpu().numpy())
    print('mean SAD:', round(SAD.mean().item()*100,2))
    print('mean RMSE:', round(RMSE.mean().item()*100,2))
    # savemat(file + '.mat', {'E':ini.transpose()})
    # savemat(file + '_proposed.mat', {'E_proposed':net_decoder.layer.weight[ind].detach().cpu().numpy().transpose(1,0)})
    SAD_list.append(SAD.mean().item())
    RMSE_list.append(RMSE.mean().item())
SAD_times_array = np.array(SAD_list)
RMSE_times_array = np.array(RMSE_list)
print('SAD_times_mean:{}, std:{}'.format(round(SAD_times_array.mean()*100,4), round(SAD_times_array.std()*100,2)))
print('RMSE_times_mean:{}, std:{}'.format(round(RMSE_times_array.mean()*100, 4), round(RMSE_times_array.std()*100,2)))


