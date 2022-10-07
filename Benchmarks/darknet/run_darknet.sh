#!/bin/bash

pids=""
for i in {1..1}; do
#for i in {1..2}; do
#for i in {1..5}; do
#for i in {1..10}; do
#for i in {1..20}; do
#for i in {1..50}; do
    
    # training a classifier on cifar 
    # with 20 of these running, I got 0 failures, but they took 38 minutes
    # usually 2 min 30s, i think
    #(time ./darknet classifier train cfg/cifar.data cfg/cifar_small.cfg) &

    # you only look once detection
    # this is usually 1s to run
    # with 20, sometimes a few fail; they take about 8s.
    #(time ./darknet detect cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights data/dog.jpg) &

    # predicting with a classifier on a picture of a dog
    # this is a 1.7s run
    # with 20 of them going, lots of fails. never got them all to go through
    # 10 of them complete in 4.3s
    (time ./darknet classifier predict cfg/imagenet1k.data cfg/darknet19.cfg weights/darknet19.weights data/dog.jpg) &

    # generating text with an rnn trained on shakespeare's works
    # with 20 processes, i saw no fails on my first run.
    # they took 14s
    #(time ./darknet rnn generate cfg/rnn.cfg weights/shakespeare.weights -srand 0) &



    #
    # densenet (tested this one by accident)
    #
    # 35s
    #(time cat image-names-medium.txt | ./darknet classifier predict cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights) &
    # 1min 10s
    #(time cat image-names-large.txt | ./darknet classifier predict cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights) &


    #
    # classifier prediction
    #
    # 31s    
    #(time cat image-names-medium.txt | ./darknet classifier predict cfg/imagenet1k.data cfg/darknet19.cfg weights/darknet19.weights) &
    # 1min
    #(time cat image-names-large.txt | ./darknet classifier predict cfg/imagenet1k.data cfg/darknet19.cfg weights/darknet19.weights) &


    # 
    # yolo
    #
    # 1min 5s
    #(time cat image-names-medium.txt | ./darknet detect cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights) &
    # 2min 10s
    #(time cat image-names-large.txt | ./darknet detect cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights) &


    #
    # rnn
    #
    # 5s
    #(time ./darknet rnn generate cfg/rnn.cfg weights/shakespeare.weights -len 10000) &
    # 53s
    #(time ./darknet rnn generate cfg/rnn.cfg weights/shakespeare.weights -len 100000) &


    # 
    # train classifier on cifar
    # 
    # 2min 30s
    #(time ./darknet classifier train cfg/cifar-cporter.data cfg/cifar_small.cfg) &




    # store PID of process
    pids+=" $!"
done

# Wait for all processes to finish and dump success return value
for p in $pids; do
    if wait $p; then
        echo "Process $p success"
    else
        echo "Process $p fail"
    fi
done
