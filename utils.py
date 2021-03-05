from scipy.io import savemat
from easydict import EasyDict as edict
from net_factory import *
import matplotlib.pyplot as plt
import torch.optim as optim
from DmaxD import *
hp = edict()


def show_unmix_result(hp_total_num, a, endmember_pre, show='False'):
    if show == True:
        for i in range(hp_total_num):
            fig = plt.gcf()
            fig.set_size_inches(7.0 / 3, 7.0 / 3)  # dpi = 300, output = 700*700 pixels
            plt.gca().xaxis.set_major_locator(plt.NullLocator())

            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

            plt.margins(0, 0)
            plt.imshow(a.detach().cpu().numpy()[:, :, i], cmap=plt.cm.gray)
            # plt.savefig(file + str(i) +'.png',transparent=True, dpi=300, pad_inches = 0)
            plt.show()

        plt.figure(1)
        plt.plot(endmember_pre.detach().cpu().numpy())
        plt.figure(2)
        plt.show()