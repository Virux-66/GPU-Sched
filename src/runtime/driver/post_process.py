#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import os



def get_experiment_time(workloader_log):
    with open( RESULT_PATH + '/'+workloader_log,'r') as f:
        lines=f.readlines()
        for one_line in lines:
            if 'failed' in one_line or 'FAILED' in one_line:
                return None
            if 'TOTAL_EXPERIMENT_TIME' in one_line:
                splited_list=one_line.split()
                return (float)(splited_list[3])
                
def plot_figure(workloads_data):
    labels=[]       #['w0','w1',..,'w6']
    cols=[]         #contain #algorithm list [single-assignment,zero,mgb_basic,ai-mgb_basic]
    #print(list(workload_data.keys()))
    alg_num=len(algorithms) #how many algorithm are compared

    for i in range(alg_num):
        cols.append([])

    for k in workload_data.keys():
        labels.append(k)
     
    for i in range(len(workload_data.keys())):
        key='w'+str(i)
        for _alg in workload_data[key].keys():
            if _alg == 'single-assignment':
                cols[0].append(workload_data[key][_alg])
            elif _alg == 'zero':
                cols[1].append(workload_data[key][_alg])
            elif _alg == 'mgb_basic':
                cols[2].append(workload_data[key][_alg]) 
            elif _alg == 'ai-mgb_basic':
                cols[3].append(workload_data[key][_alg])
    
    x = np.arange(len(labels))
    width=0.2  #the width of the bars
    fig, ax = plt.subplots(figsize=(6,5))
    rects1 = ax.bar(x-3*width/2,cols[0],width,label='single-assignment')   #single-assignment
    rects2 = ax.bar(x-width/2,cols[1],width,label='zero')
    rects3 = ax.bar(x+width/2,cols[2],width,label='mgb_basic')
    rects4 = ax.bar(x+3*width/2,cols[3],width,label='ai-mgb_basic')

    ax.set_ylabel('Time(s)')
    ax.set_title('on the 3080Ti System')
    ax.set_xticks(x,labels)
    ax.legend()

    ax.bar_label(rects1,padding=3)
    ax.bar_label(rects2,padding=3)
    ax.bar_label(rects3,padding=3)
    ax.bar_label(rects4,padding=3)
    #fig.tight_layout(pad=0)

    plt.savefig("figure1.png")
    plt.show()
    #for i in range(len(cols)):
        #print(cols[i])


    #plt.ylabel('Time(s)')
    #plt.title('on the 3080Ti system')
    

algorithms=[
    'zero',
    'single-assignment',
    'mgb_basic',
    'ai-mgb_basic'
]

workload_data={
    'w0':{},
    'w1':{},
    'w2':{},
    'w3':{},
    'w4':{},
    'w5':{},
    'w6':{}
}

RESULT_PATH='./results-12.23-17:26-2G'

if __name__ == '__main__':
    files_list=os.listdir(RESULT_PATH)
    resultfile_dictionary={}

    for alg in algorithms:  # 4 algorithms in total
        resultfile_dictionary[alg]=[]

    for file_name in files_list:
        sublist=file_name.split('.')
        resultfile_dictionary[sublist[1]].append(file_name)

    for alg in resultfile_dictionary.keys():
        for f_name in resultfile_dictionary[alg]:
            workload_prefix=f_name.split('.')[0][-1]
            workload_index='w'+workload_prefix
            if 'workloader-log' in f_name:
                epsilon= get_experiment_time(f_name)
                workload_data[workload_index][alg]=epsilon

    #print(workload_data)
    #for it in workload_data.items():
        #print(it)
    plot_figure(workload_data)    


    #print(result_directory)


