import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_train_test_acc(data_path=None, header=None,
                            save_path=None):
    df = pd.read_csv(data_path,sep=',',header=header)
    plt.plot(df['train_acc'].to_list(), label='train accuracy')
    plt.plot(df['test_acc'].to_list(), label='test accuracy')
    plt.ylabel('Accuracy'); plt.xlabel('Epochs')
    plt.legend(loc="upper left")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path + '/train_test_acc.jpeg')

def get_avg(data_paths=None, header=None):
    assert(len(data_paths) == len(header))
    #create an empty list of size 200
    #keeps the avg of 'test_acc' for each epoch
    avg_list = [0] * 200
    for i, data in enumerate(data_paths):
        df = pd.read_csv(data,sep=',',header=header[i])
        #sum the test_acc for each epoch
        avg_list = [sum(x) for x in zip(avg_list, df['test_acc'][:-1].to_list())]
    #divide each element of the list by the number of data_paths
    avg_list = [x / len(data_paths) for x in avg_list]
    return avg_list

def plot_multi(avgs,
               data_names=None,
               save_path=None):
    # assert(len(data_path) == len(header))
    for i, data in enumerate(avgs):
        if data_names[i]:
            plt.plot(data, label=data_names[i])
        else:
            plt.plot(data)
    # plt.yticks(np.arange(int(min(min(avgs, key=min))), 101, 5))
    ####plot all epochs
    plt.yticks(np.arange(35, 101, 5))
    plt.xticks(np.arange(0, 201, 20))

    ####plot last 40 epochs
    # plt.yticks(np.arange(35, 101, 0.1))
    # plt.xticks(np.arange(0, 201, 10))
    # plt.xlim(160, 201)
    # plt.ylim(95.7, 96.1)
    ####

    plt.ylabel('Test Accuracy'); plt.xlabel('Epochs')
    plt.legend(loc="lower right")
    if save_path[-5:] == '.jpeg':
        lp = save_path.rfind('/')
        if lp != -1:
            Path(save_path[:lp]).mkdir(parents=True, exist_ok=True)                                         
        plt.savefig(save_path)
    else:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path + '/mngvscut.jpeg')

# plot_train_test_acc(data_path='rand_logs/MANGO/OrigMANGO-1ph-faster-s0.csv',
#           header=28,
#           save_path='plot1.jpeg')

mng_faredge = get_avg(data_paths = ['mango_logs/mng_c10_res18_faredge_s0.csv',
                                    'mango_logs/mng_c10_res18_faredge_s1.csv',
                                    'mango_logs/mng_c10_res18_faredge_s2.csv',
                                    'mango_logs/mng_c10_res18_faredge_s3.csv',
                                    'mango_logs/mng_c10_res18_faredge_s4.csv',],
                                    header = [22, 22, 22, 22, 22],)
mng_randcolor = get_avg(data_paths=['mango_logs/mng_c10_res18_randcolor_s0.csv',
                                    'mango_logs/mng_c10_res18_randcolor_s1.csv',
                                    'mango_logs/mng_c10_res18_randcolor_s2.csv',
                                    'mango_logs/mng_c10_res18_randcolor_s3.csv',
                                    'mango_logs/mng_c10_res18_randcolor_s4.csv'],
                                    header=[22, 22, 22, 22, 22])
mng_3x3_avg = get_avg(data_paths=['mango_logs/mng_c10_res18_3x3_s0.csv',
                                 'mango_logs/mng_c10_res18_3x3_s1.csv',
                                 'mango_logs/mng_c10_res18_3x3_s2.csv',
                                 'mango_logs/mng_c10_res18_3x3_s3.csv',
                                 'mango_logs/mng_c10_res18_3x3_s4.csv',],
                                 header=[22, 22, 22, 22, 22])
mng_avg = get_avg(data_paths=['rand_logs/MANGO/OrigMANGO-1ph-faster-s0.csv',
                    'rand_logs/MANGO/OrigMANGO-1ph-faster-s1.csv',
                    'rand_logs/MANGO/OrigMANGO-1ph-faster-s2.csv',
                    'rand_logs/MANGO/OrigMANGO-1ph-faster-s3.csv',
                    'rand_logs/MANGO/OrigMANGO-1ph-faster-s5.csv'],
                    header=[28,28,28,28,28])
cutout_avg = get_avg(data_paths=['rand_logs/MANGO/cutout-c10-s0.csv',
                        'rand_logs/MANGO/cutout-c10-s1.csv',
                        'rand_logs/MANGO/cutout-c10-s2.csv',
                        'rand_logs/MANGO/cutout-c10-s3.csv',
                        'rand_logs/MANGO/cutout-c10-s4.csv'],
                        header=[28,28,28,28,28])
#concat mng_avg and cutout_avg as a list of two lists
# avgs = [cutout_avg, mng_faredge, mng_randcolor, mng_3x3_avg, mng_avg]
avgs = [cutout_avg, mng_avg]
plot_multi(avgs,
            # data_names=['Cutout', 'Faredge', 'Rand-Per-Pixel', 'MANGO with N = 9', 'Original MANGO'],
           data_names=['Cutout', 'Original MANGO'],
           save_path='plots/')