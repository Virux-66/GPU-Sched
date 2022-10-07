#!/usr/bin/env python3.6

import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')

__author__ = 'Chris Porter'

ytick_fontsize = 15
xtick_fontsize = 15
title_fontsize = 30
ylabel_fontsize = 20
legend_fontsize = 20

#colors to use, if needed
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),                        
                (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),                      
                (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),                    
                (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]                      

def drawSideBars(x_var, y_multiVar, y_varLegends, ImgOutName, Title, Ylabel):
    """Draws the graph
    In: x_var  y_multiVar  y_varLegends ImgOutName  Title Ylabel
    Out: Image with bargraph comparison
    The function takes in "x variables (x_var)" and multiple y_var to be stacked 
    and outputs an image with name ImgoutName, title Title, and ylabel Ylabel
    """
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.                 
    for i in range(len(tableau20)):
            r, g, b = tableau20[i]
            tableau20[i] = (r / 255., g / 255., b / 255.) 

    #different hatches
    patterns = ('.', '+', 'x', '\\', '*', 'o', 'O', '.')
    num_patterns = len(patterns)
    fig, ax = plt.subplots()

    index = None
    margin = 0.1
    width = (1.0 - 2*margin)/len(y_multiVar)

    rects = []
    index = np.arange(len(x_var))
    
    prev = None
    i = 0
    for y_var in y_multiVar:#iterate over each values, creating a rect with bottom assigned
        print(y_var)
        print(index)
        rects.append(ax.bar(index + i*width, y_var, width, color= 'white', hatch = 2*patterns[i%num_patterns]))
        i += 1
       
    ax.set_ylabel(Ylabel, fontsize=ylabel_fontsize)
    #ax.set_title(Title, fontsize=title_fontsize)
    ax.set_xticks(index + width)
    ax.set_xticklabels(x_var, fontsize=xtick_fontsize)

    #ax.set_xlim(-width)
    ax.set_xlim([-0.2,9])

    ax.set_ylim([0.8,1.25])
    #ax.set_ylim([0,10])


    #the_y = y_multiVar[2][14]
    idx = 0
    for i, v in enumerate(y_multiVar[2]):
        #ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
        idx += 1
        if idx == 14:
            the_i = i
            the_v = v
    idx = 0
    #rects = ax.patches
    #print len(ax.patches)
    #labels = ["label%d" % i for i in xrange(len(rects))]
    #for rect, label in zip(rects, labels):
    #for idx in xrange(len(ax.patches)):
    #for rect in rects:
    for rect in ax.patches:
        idx += 1
        if idx == 14:
        #if (idx+1) == 14:
            height = rect.get_height()
            # add text for 7.11x
            #ax.text(rect.get_x() + rect.get_width() / 2,
            #        height + 1.75,
            #        #label,
            #        #str(the_v),
            #        "7.11x",
            #        ha='center',
            #        va='bottom')

    #axis background
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    #legends
    ax.legend(rects, y_varLegends, loc=(0.7,0.75), fontsize=legend_fontsize)
    #ax.legend(rects, y_varLegends, fontsize=legend_fontsize)

    plt.xticks(rotation=-45)
    plt.yticks(fontsize=ytick_fontsize)
    plt.tight_layout()
    plt.show()
    #plt.savefig(ImgOutName)#This is not very useful because Aspect ratio cannot be adjusted


# TODO: will need to deal with range of y axis. see "ax.set_ylim([0,4])" above
if len(sys.argv) is not 2:
    print('Usage: pass "throughput.in" or "slowdown.in" to select which graph to make')

x_var = []
y_multiVar = [[], [], []]
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip().split()
        x_var.append(line[0]) # workload
        y_multiVar[0].append(float(line[1])) # normalized sa
        y_multiVar[1].append(float(line[2])) # cg
        y_multiVar[2].append(float(line[3])) # mgb

y_varLegends = [ "sa", "mgb-kernels", "mgb-data-xfers" ]
ImgOutName = "throughput_improvement.png"
Title = ""
Ylabel = "Normalized Slowdown"
drawSideBars(x_var, y_multiVar, y_varLegends, ImgOutName, Title, Ylabel)
