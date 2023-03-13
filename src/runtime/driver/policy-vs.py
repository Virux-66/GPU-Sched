#!/bin/python3

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tabulate
import numpy as np
import csv
import os
import sys

FIGURE_FORMAT='.png' #png or pdf

def usage_and_exit():
    print()
    print('This is a python script used to compared the block-level-guided\
          and the kernel-level-guided policy')
    print(' Usage:')
    print('     {} <block result> <kernel result>'.format(sys.argv[0]))
    sys.exit(1)

def plot_global_setting():
    plt.rcParams['figure.dpi']=300
    plt.rcParams['savefig.dpi']=300
    plt.rcParams['font.family']='Times New Roman'

if __name__=='__main__':
    if len(sys.argv)<3:
        usage_and_exit()
    plot_global_setting()
    block_result_dir=sys.argv[1]
    kernel_result_dir=sys.argv[2]
    block_result_filelist=os.listdir(block_result_dir)
    kernel_result_filelist=os.listdir(kernel_result_dir)
    aimgb_block=[]
    aimgb_kernel=[]

    for f in block_result_filelist:
        alg=f.split('.')[1]
        suffix=f.split('.')[-1]
        if alg == 'ai-mgb_basic' and suffix=='workloader-log':
            aimgb_block.append(f)        

    for f in kernel_result_filelist:
        alg=f.split('.')[1]
        suffix=f.split('.')[-1]
        if alg == 'ai-mgb_basic' and suffix=='workloader-log':
            aimgb_kernel.append(f)
    block_time=9*[0]
    kernel_time=9*[0]

    for fname in aimgb_block:
        with open(sys.argv[1]+'/'+fname) as f:
            lines=f.readlines()
            widx=int((fname.split('.')[0]).split('_')[-1])-1
            for one_line in lines:
                if 'failed' in one_line or 'FAILED' in one_line:
                    print('A workload fails completion')
                    exit(1)
                if 'TOTAL_EXPERIMENT_TIME' in one_line:
                    block_time[widx]=32/float(one_line.split()[-1])

    for fname in aimgb_kernel:
        with open(sys.argv[2]+'/'+fname) as f:
            lines=f.readlines()
            widx=int((fname.split('.')[0]).split('_')[-1])-1
            for one_line in lines:
                if 'failed' in one_line or 'FAILED' in one_line:
                    print('A workload fails completion')
                    exit(1)
                if 'TOTAL_EXPERIMENT_TIME' in one_line:
                    kernel_time[widx]=(32/float(one_line.split()[-1]))/block_time[widx]
    sum_time=0    
    for i in range(8):
        sum_time+=kernel_time[i]
    kernel_time[8]=sum_time/8

    labels=['W1','W2','W3','W4','W5','W6','W7','W8','Avg'] 
    x=np.arange(len(labels))
    width=0.3
    fig, ax=plt.subplots(figsize=(8,5))
    rects1  = ax.bar(x-width/2,[1]*9,width,label='BLG',color='#358b9c')
    rects2  = ax.bar(x+width/2,kernel_time,width,label='KLG',color='#5ca781')
    ax.set_xlabel('Workloads')
    ax.set_ylabel('Normalized Throughput')
    ax.set_xticks(x,labels)
    ax.set_yticks([0,0.5,1,1.5])
    ax.legend(loc='upper right')
    plt.savefig('./policy-comparison'+FIGURE_FORMAT)
    plt.show()