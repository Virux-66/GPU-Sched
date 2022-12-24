#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import csv
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
                
def plot_throughput_figure(workloads_data):
    plt.rcParams['figure.dpi']=300
    plt.rcParams['savefig.dpi']=300
    labels=[]       #['w0','w1',..,'w6', 'avg']
    cols=[]         #contain #algorithm list [single-assignment,zero,mgb_basic,ai-mgb_basic]
    #print(list(workload_data.keys()))
    alg_num=len(algorithms) #how many algorithm are compared
    jobs_num_per_workload=32

    for i in range(alg_num):
        cols.append([])

    for k in workload_data.keys():
        labels.append(k)
    labels.append('Avg')
     
    for i in range(len(workload_data.keys())):
        key='W'+str(i)
        for _alg in workload_data[key].keys():
            if _alg == 'single-assignment':
                cols[0].append(workload_data[key][_alg])
            elif _alg == 'zero':
                cols[1].append(workload_data[key][_alg])
            elif _alg == 'mgb_basic':
                cols[2].append(workload_data[key][_alg]) 
            elif _alg == 'ai-mgb_basic':
                cols[3].append(workload_data[key][_alg])
    
    #calculate throughput(jobs/second) and normalization
    for alg_index in range(alg_num):
        cols[alg_index]=[jobs_num_per_workload/t for t in cols[alg_index]]
    for alg_index in range(1,alg_num):
        for jndex in range(len(cols[alg_index])):
            cols[alg_index][jndex]/=cols[0][jndex]
    cols[0]=[1]*len(cols[0])    #baseline

    #calculate average throughput improvement 
    for alg_index in range(alg_num):
        avg_imp=sum(cols[alg_index])/len(cols[alg_index])
        cols[alg_index].append(avg_imp)

    x = np.arange(len(labels))
    width=0.2  #the width of the bars
    fig, ax = plt.subplots(figsize=(6,5))
    rects1 = ax.bar(x-3*width/2,cols[0],width,label='single-assignment')   #single-assignment
    rects2 = ax.bar(x-width/2,cols[1],width,label='zero')
    rects3 = ax.bar(x+width/2,cols[2],width,label='mgb_basic')
    rects4 = ax.bar(x+3*width/2,cols[3],width,label='ai-mgb_basic')

    ax.set_xlabel('Workloads')
    ax.set_ylabel('Normalized Throughput')
    ax.set_title('on the 3080Ti System')
    ax.set_xticks(x,labels)
    ax.legend()

    #ax.bar_label(rects1,padding=3)
    #ax.bar_label(rects2,padding=3)
    #ax.bar_label(rects3,padding=3)
    #ax.bar_label(rects4,padding=3)
    #fig.tight_layout(pad=0)

    plt.savefig("normalized_throughput_figure.jpg")
    plt.show()

def plot_gpu_utilization_figure(path,w_index=6,device_index=0): #Default, use the last one workload to observe gpu utilization improvement
    plt.rcParams['figure.dpi']=300
    plt.rcParams['savefig.dpi']=300

    files_list=os.listdir(path)
    alg_gpu_utilization_dict={}
    for f in files_list:
        f_w_index=(((f.split('.'))[0]).split('_'))[2]
        file_suffix=(f.split('-'))[-1]
        if (int)(f_w_index) == w_index and file_suffix == 'sched_gpu.csv':
            _alg=f.split('.')[1]
            alg_gpu_utilization_dict[_alg]=path + '/' + f
    #print(alg_gpu_utilization_dict)    

    #We have obtain all gpu utilization record files. Now we plot the figure
    time_point=[] #contain 4 list
    utilization=[] #contain 4 list
    algorithms_label=[] #4 str

    for _alg in alg_gpu_utilization_dict.keys():
        algorithms_label.append(_alg)
        time_point.append([])
        utilization.append([])
        with open(alg_gpu_utilization_dict[_alg],'r') as f:
            reader=csv.reader(f)
            reader=list(reader)
            time_point_base=(int)(reader[1][0])

            for each_row in reader:
                if each_row[0]=='timestamp':
                    continue
                signle_time_point=(int)(each_row[0])
                single_gpu_utilization=(int)(each_row[device_index+1])   #for the 0 device
                (time_point[-1]).append(signle_time_point-time_point_base)
                (utilization[-1]).append(single_gpu_utilization)
    plt.plot(time_point[0],utilization[0],'r.',\
             time_point[1],utilization[1],'b.',\
             time_point[2],utilization[2],'g.',\
             time_point[3],utilization[3],'c.')
    plt.savefig('gpu_utilization.jpg')
    plt.show()
    

algorithms=[
    'zero',
    'single-assignment',
    'mgb_basic',
    'ai-mgb_basic'
]

workload_data={
    'W0':{},
    'W1':{},
    'W2':{},
    'W3':{},
    'W4':{},
    'W5':{},
    'W6':{}
}

RESULT_PATH='./results-12.24-14:03-1.5G'

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
            workload_index='W'+workload_prefix
            if 'workloader-log' in f_name:
                epsilon= get_experiment_time(f_name)
                if epsilon == None:
                    print('Some jobs corrupt')
                    exit(1)
                workload_data[workload_index][alg]=epsilon

    plot_throughput_figure(workload_data)    
    plot_gpu_utilization_figure(RESULT_PATH)
