#!/bin/python3

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tabulate
import numpy as np
import csv
import os
import sys


def usage_and_exit():
    print()
    print(' Usage:')
    print('     {} <resolved path> <visualized path>'.format(sys.argv[0]))
    exit(1)

def plot_global_setting():
    plt.rcParams['figure.dpi']=300
    plt.rcParams['savefig.dpi']=300
    plt.rcParams['font.family']='Times New Roman'


if __name__ == '__main__':
    if(len(sys.argv)<3):
        usage_and_exit()
    plot_global_setting()
    motivation_path=sys.argv[1]
    visualized_path=sys.argv[2]
    execution_time=[]
    flist=os.listdir(motivation_path)
    file_list=[]
    for _f in flist:
        if  'workloader-log' in _f and 'zero' in _f:
            file_list.append(_f)
    file_list.sort()

    for f_name in file_list:
        with open(motivation_path+'/'+f_name) as f:
            lines=f.readlines()
            for each_line in lines:
                if 'TOTAL_EXPERIMENT_TIME' in each_line:
                    execution_time.append(each_line.split()[3])
    #print(execution_time)
    table=[['Benchmarks','Execution Time'],
           ['BinomialOptions',str(round(float(execution_time[2]),4))],
           ['2xBinomialOptions',str(round(float(execution_time[3]),4))],
           ['Scan',str(round(float(execution_time[0]),4))],
           ['2xScan', str(round(float(execution_time[1]),4))],
           ['Scan+BinomialOptions',str(round(float(execution_time[4]),4))]]

    print(tabulate.tabulate(table,headers='firstrow',tablefmt='fancy_grid',numalign='center'))
    
    with open(visualized_path+'/motivation_table','w') as _f:
        _f.write(tabulate.tabulate(table,headers='firstrow',tablefmt='fancy_grid',numalign='center'))



