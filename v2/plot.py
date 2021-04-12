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

def plot_multi(data_path=None, header=None,
               data_names=None,
               save_path=None):
    assert(len(data_path) == len(header))
    for i, data in enumerate(data_path):
        df = pd.read_csv(data,sep=',',header=header[i])
        if data_names[i]:
            plt.plot(df['test_acc'].to_list(), label=data_names[i])
        else:
            plt.plot(df['test_acc'].to_list())
    plt.ylabel('Test Accuracy'); plt.xlabel('Epochs')
    plt.legend(loc="lower right")
    if save_path[-5:] == '.jpeg':
        lp = save_path.rfind('/')
        Path(save_path[:lp]).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path + '/multi_acc.jpeg')

# plot_train_test_acc(data_path='logs/MANGO/orignewcutout.csv',
#           header=23,
#           save_path='test')

plot_multi(data_path=['logs/MANGO/orignewcutout.csv', 'logs/MANGO/wide-wide-3x3.csv'], 
           header=[23, 23],
           data_names=['MANGO', 'MANGO 3x3'], 
           save_path='test.jpeg')