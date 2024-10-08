import tensorflow as tf
#keras = tf.keras
from tensorflow import keras
#from keras import layers
##from tensorflow.keras import ops
##from tensorflow.keras import utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
#from keras.optimizers import SGD
#from keras.datasets import mnist
##from tensorflow.keras.utils import PyDataset
from keras import backend #as K
#from keras.backend import clear_session

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
# from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import classification_report


import matplotlib.pyplot as plt
import numpy as np
import argparse

import gc


import concurrent.futures # threadpoolexecutor
import copy # deepcopy
import traceback # debugging threads
import time # localtime to unix time
import psutil # System parameter (CPU usage, RAM usage)
import sys # RAM usage of lists, exit()
import random # randomise tracks by withe noise and reelementing

#np.set_printoptions(threshold=sys.maxsize) # change limit of printed numbers in array
#tboard_callback = tf.keras.callbacks.TensorBoard(histogram_freq = 1,
#                                                 profile_batch = '500,520')

# surpress warnings for better output readability \n",
import warnings
warnings.filterwarnings('ignore')

#tf.config.list_physical_devices('GPU')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # deactivate GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     # deactivate terminal warning KERAS - Tenserflow




def gpu_mem_limits():
    pass

#import ast
def mainKNN_T(path, file):
    print('Startzeit:\t', time.ctime(time.time()))

    startTime = time.time() # timestemp at start
    preTime = time.time() # timestemp at the previous function


    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #print("Num GPUs Avalible: ", len(physical_devices))
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # 1* Dataset > Y * Track > X * Position > 5/6 * Element

    parentTrackLengthInPositions = 100 # Number of elements per track of the input .csv
    elementsPerPositionFile = 6 # Number of values per element (MMSI, Timestamp, x [m], y [m], deg [°], r [m])#5 Number of values per element (timestamp, x [m], y [m], deg [°], r [m])
    childrenTrackLengthInPositions = tuple([3, 4, 5, 6, 7, 8, 9, 12, 15]) # Number of positions per children track
    sigma = tuple([0, 5, 10, 15, 20, 25, 30, 35, 40]) # random spreat of the x,y Position in [m]
    scalefactor = 10000 # factor by which all XY coordinates are divided. The factor is usually to get max value to 1. 10000 m is the highest possible value.
    testProcentage = 10 #tuple([5, 10, 15, 20, 25, 30, 35, 40, 45]) # 10 Amongth of testdata in %
    precisionInDigits = 1 # Precision of target values in digets
    unitS = tuple([3, 3, 3, 5, 5, 5, 7, 7, 7]) # units = notes = Neuronen
    batchS = tuple([8, 16, 32, 48, 64, 96, 128, 192, 256]) #64 # amonght of data used in one testrun
    epochS = tuple([5, 10, 25, 50, 100, 250, 500, 750, 1000]) #5number of training learing runs
    layerS = tuple([1, 1, 1, 2, 2, 2, 3, 3, 3]) # layers of the knn
    offset = 1 # offset of nodes
    maxParentTracks = 3500000 # number of tracks that are used for the FFN
    elementsPerPositionFFN = 5 # Timestamp [s], x [m], y [m], deg [°], r [m]
    activationFunction = tuple(['elu', 'gelu', 'leaky_relu', 'linear', 'mish', 'relu','sigmoid', 'selu', 'softmax']) #, 'relu'
    numberOfRunsPerConfig = 1 #5
    optimiZer = tuple(['Adadelta', 'Adafactor', 'Adam', 'Adamax', 'Ftrl', 'Lion', 'Nadam', 'RMSprop', 'SGD']) # ['SGD', 'RMSprop', 'Adam', 'AdamW', 'Adadelta', 'Adagrad', 'Adamax', 'Adafactor', 'Nadam'])
    learningRate = tuple([0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1])
    dropoutRate = tuple([0, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5])  #0

    # parameterization of the FFN
    parameters = [0]*19

    hyperParameters = [0]*19

    
    
    # batchs = len(scaledTrainingMeas) / epoch
    parameters[0] = parentTrackLengthInPositions
    parameters[1] = elementsPerPositionFile
    hyperParameters[2] = childrenTrackLengthInPositions
    hyperParameters[3] = sigma
    parameters[4] = scalefactor
    parameters[5] = testProcentage #parameters #hyperParameters
    parameters[6] = precisionInDigits
    hyperParameters[7] = unitS
    hyperParameters[8] = batchS
    hyperParameters[9] = epochS
    hyperParameters[10] = layerS
    parameters[11] = offset
    parameters[12] = maxParentTracks
    parameters[13] = elementsPerPositionFFN
    hyperParameters[14] = activationFunction
    parameters[15] = numberOfRunsPerConfig
    hyperParameters[16] = optimiZer
    hyperParameters[17] = learningRate
    hyperParameters[18] = dropoutRate #hyperParameters #parameters
    

    taguchiL81=[[0]*10]*81
    
    taguchiL81[	0	] = 	[0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	0	,	 0]
    taguchiL81[	1	] = 	[0	,	1	,	2	,	3	,	4	,	5	,	6	,	7	,	8	,	 1]
    taguchiL81[	2	] = 	[0	,	2	,	1	,	6	,	8	,	7	,	3	,	5	,	4	,	 2]
    taguchiL81[	3	] = 	[0	,	3	,	6	,	7	,	1	,	4	,	5	,	8	,	2	,	 3]
    taguchiL81[	4	] = 	[0	,	4	,	8	,	1	,	5	,	6	,	2	,	3	,	7	,	 4]
    taguchiL81[	5	] = 	[0	,	5	,	7	,	4	,	6	,	2	,	8	,	1	,	3	,	 5]
    taguchiL81[	6	] = 	[0	,	6	,	3	,	5	,	2	,	8	,	7	,	4	,	1	,	 6]
    taguchiL81[	7	] = 	[0	,	7	,	5	,	8	,	3	,	1	,	4	,	2	,	6	,	 7]
    taguchiL81[	8	] = 	[0	,	8	,	4	,	2	,	7	,	3	,	1	,	6	,	5	,	 8]
    taguchiL81[	9	] = 	[1	,	0	,	2	,	7	,	6	,	8	,	4	,	3	,	5	,	 2]
    taguchiL81[	10	] = 	[1	,	1	,	1	,	1	,	1	,	1	,	1	,	1	,	1	,	 0]
    taguchiL81[	11	] = 	[1	,	2	,	0	,	4	,	5	,	3	,	7	,	8	,	6	,	 1]
    taguchiL81[	12	] = 	[1	,	3	,	8	,	5	,	7	,	0	,	6	,	2	,	4	,	 5]
    taguchiL81[	13	] = 	[1	,	4	,	7	,	8	,	2	,	5	,	3	,	6	,	0	,	 3]
    taguchiL81[	14	] = 	[1	,	5	,	6	,	2	,	3	,	7	,	0	,	4	,	8	,	 4]
    taguchiL81[	15	] = 	[1	,	6	,	5	,	0	,	8	,	4	,	2	,	7	,	3	,	 8]
    taguchiL81[	16	] = 	[1	,	7	,	4	,	3	,	0	,	6	,	8	,	5	,	2	,	 6]
    taguchiL81[	17	] = 	[1	,	8	,	3	,	6	,	4	,	2	,	5	,	0	,	7	,	 7]
    taguchiL81[	18	] = 	[2	,	0	,	1	,	5	,	3	,	4	,	8	,	6	,	7	,	 1]
    taguchiL81[	19	] = 	[2	,	1	,	0	,	8	,	7	,	6	,	5	,	4	,	3	,	 2]
    taguchiL81[	20	] = 	[2	,	2	,	2	,	2	,	2	,	2	,	2	,	2	,	2	,	 0]
    taguchiL81[	21	] = 	[2	,	3	,	7	,	0	,	4	,	8	,	1	,	5	,	6	,	 4]
    taguchiL81[	22	] = 	[2	,	4	,	6	,	3	,	8	,	1	,	7	,	0	,	5	,	 5]
    taguchiL81[	23	] = 	[2	,	5	,	8	,	6	,	0	,	3	,	4	,	7	,	1	,	 3]
    taguchiL81[	24	] = 	[2	,	6	,	4	,	7	,	5	,	0	,	3	,	1	,	8	,	 7]
    taguchiL81[	25	] = 	[2	,	7	,	3	,	1	,	6	,	5	,	0	,	8	,	4	,	 8]
    taguchiL81[	26	] = 	[2	,	8	,	5	,	4	,	1	,	7	,	6	,	3	,	0	,	 6]
    taguchiL81[	27	] = 	[3	,	0	,	6	,	8	,	5	,	2	,	1	,	7	,	4	,	 6]
    taguchiL81[	28	] = 	[3	,	1	,	8	,	2	,	6	,	4	,	7	,	5	,	0	,	 7]
    taguchiL81[	29	] = 	[3	,	2	,	7	,	5	,	1	,	6	,	4	,	0	,	8	,	 8]
    taguchiL81[	30	] = 	[3	,	3	,	3	,	3	,	3	,	3	,	3	,	3	,	3	,	 0]
    taguchiL81[	31	] = 	[3	,	4	,	5	,	6	,	7	,	8	,	0	,	1	,	2	,	 1]
    taguchiL81[	32	] = 	[3	,	5	,	4	,	0	,	2	,	1	,	6	,	8	,	7	,	 2]
    taguchiL81[	33	] = 	[3	,	6	,	0	,	1	,	4	,	7	,	8	,	2	,	5	,	 3]
    taguchiL81[	34	] = 	[3	,	7	,	2	,	4	,	8	,	0	,	5	,	6	,	1	,	 4]
    taguchiL81[	35	] = 	[3	,	8	,	1	,	7	,	0	,	5	,	2	,	4	,	6	,	 5]
    taguchiL81[	36	] = 	[4	,	0	,	8	,	3	,	2	,	7	,	5	,	1	,	6	,	 8]
    taguchiL81[	37	] = 	[4	,	1	,	7	,	6	,	3	,	0	,	2	,	8	,	5	,	 6]
    taguchiL81[	38	] = 	[4	,	2	,	6	,	0	,	7	,	5	,	8	,	3	,	1	,	 7]
    taguchiL81[	39	] = 	[4	,	3	,	5	,	1	,	0	,	2	,	7	,	6	,	8	,	 2]
    taguchiL81[	40	] = 	[4	,	4	,	4	,	4	,	4	,	4	,	4	,	4	,	4	,	 0]
    taguchiL81[	41	] = 	[4	,	5	,	3	,	7	,	8	,	6	,	1	,	2	,	0	,	 1]
    taguchiL81[	42	] = 	[4	,	6	,	2	,	8	,	1	,	3	,	0	,	5	,	7	,	 5]
    taguchiL81[	43	] = 	[4	,	7	,	1	,	2	,	5	,	8	,	6	,	0	,	3	,	 3]
    taguchiL81[	44	] = 	[4	,	8	,	0	,	5	,	6	,	1	,	3	,	7	,	2	,	 4]
    taguchiL81[	45	] = 	[5	,	0	,	7	,	1	,	8	,	3	,	6	,	4	,	2	,	 7]
    taguchiL81[	46	] = 	[5	,	1	,	6	,	4	,	0	,	8	,	3	,	2	,	7	,	 8]
    taguchiL81[	47	] = 	[5	,	2	,	8	,	7	,	4	,	1	,	0	,	6	,	3	,	 6]
    taguchiL81[	48	] = 	[5	,	3	,	4	,	8	,	6	,	7	,	2	,	0	,	1	,	 1]
    taguchiL81[	49	] = 	[5	,	4	,	3	,	2	,	1	,	0	,	8	,	7	,	6	,	 2]
    taguchiL81[	50	] = 	[5	,	5	,	5	,	5	,	5	,	5	,	5	,	5	,	5	,	 0]
    taguchiL81[	51	] = 	[5	,	6	,	1	,	3	,	7	,	2	,	4	,	8	,	0	,	 4]
    taguchiL81[	52	] = 	[5	,	7	,	0	,	6	,	2	,	4	,	1	,	3	,	8	,	 5]
    taguchiL81[	53	] = 	[5	,	8	,	2	,	0	,	3	,	6	,	7	,	1	,	4	,	 3]
    taguchiL81[	54	] = 	[6	,	0	,	3	,	4	,	7	,	1	,	2	,	5	,	8	,	 3]
    taguchiL81[	55	] = 	[6	,	1	,	5	,	7	,	2	,	3	,	8	,	0	,	4	,	 4]
    taguchiL81[	56	] = 	[6	,	2	,	4	,	1	,	3	,	8	,	5	,	7	,	0	,	 5]
    taguchiL81[	57	] = 	[6	,	3	,	0	,	2	,	8	,	5	,	4	,	1	,	7	,	 6]
    taguchiL81[	58	] = 	[6	,	4	,	2	,	5	,	0	,	7	,	1	,	8	,	3	,	 7]
    taguchiL81[	59	] = 	[6	,	5	,	1	,	8	,	4	,	0	,	7	,	3	,	2	,	 8]
    taguchiL81[	60	] = 	[6	,	6	,	6	,	6	,	6	,	6	,	6	,	6	,	6	,	 0]
    taguchiL81[	61	] = 	[6	,	7	,	8	,	0	,	1	,	2	,	3	,	4	,	5	,	 1]
    taguchiL81[	62	] = 	[6	,	8	,	7	,	3	,	5	,	4	,	0	,	2	,	1	,	 2]
    taguchiL81[	63	] = 	[7	,	0	,	5	,	2	,	4	,	6	,	3	,	8	,	1	,	 5]
    taguchiL81[	64	] = 	[7	,	1	,	4	,	5	,	8	,	2	,	0	,	3	,	6	,	 3]
    taguchiL81[	65	] = 	[7	,	2	,	3	,	8	,	0	,	4	,	6	,	1	,	5	,	 4]
    taguchiL81[	66	] = 	[7	,	3	,	2	,	6	,	5	,	1	,	8	,	4	,	0	,	 8]
    taguchiL81[	67	] = 	[7	,	4	,	1	,	0	,	6	,	3	,	5	,	2	,	8	,	 6]
    taguchiL81[	68	] = 	[7	,	5	,	0	,	3	,	1	,	8	,	2	,	6	,	4	,	 7]
    taguchiL81[	69	] = 	[7	,	6	,	8	,	4	,	3	,	5	,	1	,	0	,	2	,	 2]
    taguchiL81[	70	] = 	[7	,	7	,	7	,	7	,	7	,	7	,	7	,	7	,	7	,	 0]
    taguchiL81[	71	] = 	[7	,	8	,	6	,	1	,	2	,	0	,	4	,	5	,	3	,	 1]
    taguchiL81[	72	] = 	[8	,	0	,	4	,	6	,	1	,	5	,	7	,	2	,	3	,	 4]
    taguchiL81[	73	] = 	[8	,	1	,	3	,	0	,	5	,	7	,	4	,	6	,	2	,	 5]
    taguchiL81[	74	] = 	[8	,	2	,	5	,	3	,	6	,	0	,	1	,	4	,	7	,	 3]
    taguchiL81[	75	] = 	[8	,	3	,	1	,	4	,	2	,	6	,	0	,	7	,	5	,	 7]
    taguchiL81[	76	] = 	[8	,	4	,	0	,	7	,	3	,	2	,	6	,	5	,	1	,	 8]
    taguchiL81[	77	] = 	[8	,	5	,	2	,	1	,	7	,	4	,	3	,	0	,	6	,	 6]
    taguchiL81[	78	] = 	[8	,	6	,	7	,	2	,	0	,	1	,	5	,	3	,	4	,	 1]
    taguchiL81[	79	] = 	[8	,	7	,	6	,	5	,	4	,	3	,	2	,	1	,	0	,	 2]
    taguchiL81[	80	] = 	[8	,	8	,	8	,	8	,	8	,	8	,	8	,	8	,	8	,	 0]



    preTime, parentDataset = CreateDataSet(startTime, preTime, path, file, parameters)
    
    
    runNumber = 0
    outerExecutor = concurrent.futures.ProcessPoolExecutor(max_workers = 4) # 4
    outerFutures = []
    
    while runNumber < len(taguchiL81): #len(taguchiL81):  
       
        parametersLocal = copy.deepcopy(parameters)
        parametersLocal[2] = hyperParameters[2][taguchiL81[runNumber][0]] # childrenTrackLengthInPositions
        parametersLocal[3] = hyperParameters[3][taguchiL81[runNumber][1]] # sigma
        #parametersLocal[5] = hyperParameters[5][taguchiL81[runNumber][2]] # testProcentage
        parametersLocal[7] = hyperParameters[7][taguchiL81[runNumber][3]] # unitS
        parametersLocal[8] = hyperParameters[8][taguchiL81[runNumber][4]] # batchS
        parametersLocal[9] = hyperParameters[9][taguchiL81[runNumber][5]] # epochS
        parametersLocal[10] = hyperParameters[10][taguchiL81[runNumber][6]] # layerS
        parametersLocal[14] = hyperParameters[14][taguchiL81[runNumber][7]] # activationFunction
        parametersLocal[16] = hyperParameters[16][taguchiL81[runNumber][8]] # optimiZer
        parametersLocal[17] = hyperParameters[17][taguchiL81[runNumber][9]] # learningRate
        parametersLocal[18] = hyperParameters[18][taguchiL81[runNumber][2]] # dropoutRate

        # start worker threads
        outerFutures.append(outerExecutor.submit(outer_thread_function, copy.copy(runNumber),  
                               copy.deepcopy(startTime), copy.deepcopy(preTime), copy.deepcopy(parentDataset), parametersLocal, copy.copy(path), copy.copy(file)))
        runNumber = runNumber + 1

    print("Main    : submitted all outerThreads to ThreadPool: " + str(runNumber) + " " + str(len(outerFutures)))
    time.sleep(1)
    
    # wait for processes to finish and collect results
    outer_results = []
    for index, future in enumerate(outerFutures):
        #print("Main    : before joining thread %d.", index)
        #while True:
            #if (future.exception() != None):
        list_of_inner_results = future.result()
        for inner_result in list_of_inner_results:
            outer_results.append(inner_result)
            #if (future.done() == True):
        print("Main    : outerThread " + str(index) + " done")
            #break
            #time.sleep(1)
                
    # all threads finished
    print("All outerThreads finished")

    #     kkn_st_path, file = path[:len(path, file)-4] + '_' + '{0:6.0f}'.format(startTime) + '_statistic.csv'
    kkn_st_path = path[:len(path) - 14] + '/AIS-KNN-Stat-Files/' + file[:len(file) - 4] + '_statistic_' + '{0:6.0f}'.format(startTime) + '_.csv'
    
    # creating a new empty file or overwriting existend file
    save_file = open(kkn_st_path, 'w')
    save_file.write((kkn_st_path + ' ' + '{0:6.0f}'.format(startTime) + '\n'))
    save_file.close()
    
    #     kkn_pr_path, file = path, file[:len(path, file)-4] + '_' + '{0:6.0f}'.format(startTime) + '_proximity.csv'
    kkn_pr_path = path[:len(path)-14] + '/AIS-KNN-Stat-Files/' + file[:len(file) - 4] + '_proximity_' +  '{0:6.0f}'.format(startTime) + '.csv'
    # creating a new empty file or overwriting existend file
    save_file = open(kkn_pr_path, 'w')
    save_file.write((kkn_pr_path + ' ' + '{0:6.0f}'.format(startTime) + '\n'))
    save_file.close()
    
    # save results
    # appending existing statistic file
    f_st = open(kkn_st_path, 'a')
    # appending existing gnu file
    f_pr = open(kkn_pr_path, 'a')
    for result in outer_results:
        f_st.write(str(result.get_state()) + '\n')
        f_pr.write(str(result.get_gnuString()) + '\n')
    f_st.close()
    f_pr.close()

    # free up memory 
    #del parentToChildTrack
    print("DONE \t" + 'Laufzeit insg: \t' + '{0:6.3f}'.format((time.time() - startTime)))
    

def outer_thread_function(runNumber, startTime, preTime, parentDataset, parameters, path, file):
    try:
        gpu_mem_limits()
        print("start outer " + str(runNumber))
        preTime, childrenTracks = parentToChildTrack(startTime, preTime, parentDataset, parameters)
        preTime, scaledTrainingMeas, scaledTrainingTarget, scaledTestMeas, scaledTestTarget = splitAndRescaleChildTrack(startTime, preTime, childrenTracks, parameters)  
        preTime, scaledTestMeas = outlierGen (startTime, preTime, scaledTestMeas,  parameters)
        preTime, scaledTrainingMeas = outlierGen (startTime, preTime, scaledTrainingMeas,  parameters)
                
        
        #if taguchiL81[runNumber][3] != taguchiL81[runNumber-1][3] or taguchiL81[runNumber][4] != taguchiL81[runNumber-1][4] or taguchiL81[runNumber][5] != taguchiL81[runNumber-1][5] or taguchiL81[runNumber][6] != taguchiL81[runNumber-1][6] or taguchiL81[runNumber][7] != taguchiL81[runNumber-1][7] or taguchiL81[runNumber][8] != taguchiL81[runNumber-1][8]: 
        print("total#" + str(runNumber) + " " + str(parameters))
        counter = 0
        innerExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=5) # 5
        innerFutures = []
        
        while counter < parameters[15]:
            #print("total#" + str(runNumber) + " " + str(counter+1) + " von " + str(parameters[15]))
            
            # start worker threads
            innerFutures.append(innerExecutor.submit(inner_thread_function, runNumber, counter, startTime, copy.copy(preTime),
                                                     scaledTrainingMeas, scaledTrainingTarget, scaledTestMeas, scaledTestTarget,
                                                     parameters, path, file))            
            counter = counter + 1

        # wait for threads to finish
        results = []
        for index, futures in enumerate(innerFutures):
            #print("Main    : before joining thread %d.", index)
            results.append(futures.result())
            #print("Main    : thread %d done", index, resultDb)
        # all threads finished
        print("end outer " + str(runNumber) + " All innerThreads finished")
        #print("Thread " + runNumber + "-" + counter + ": finished")
        return results
    except Exception as e:
        print("\n\nWorker Outer EXCEPTION: " + str(runNumber) + " : " + str(e) + "\n\n")
        traceback.print_exc()
        raise
        #sys.exit(1)


def inner_thread_function(runNumber, counter, startTime, preTime, scaledTrainingMeas, scaledTrainingTarget, scaledTestMeas, scaledTestTarget, parameters, path, file):
    try:
        gpu_mem_limits()
        print("start inner " + str(runNumber) + "-" + str(counter))
        #while True:
        #   continue
        #print("Thread " + runNumber + "-" + counter + ": starting")
        statistic, gnuString = FFN(startTime, preTime, scaledTrainingMeas, scaledTrainingTarget, scaledTestMeas, scaledTestTarget, parameters)
        state = []
        state = '; '.join(map(str,parameters))
        state = state + ('; ') + ('; '.join(map(str,statistic)))  + ('; ') + '{0:6.0f}'.format((time.time() - startTime))
        print(str(runNumber) + "-" + str(counter) + "|" + str(state))
        print("end inner " + str(runNumber) + "-" + str(counter))
        return ResultSet(runNumber, counter, state, gnuString)

    except Exception as e:
        print("\n\nWorker Inner EXCEPTION: " + str(runNumber) + "-" + str(counter) + " : " + str(e)+"\n\n")
        traceback.print_exc()
        #print(e)
        raise
        #sys.exit(1)
        



class ResultDatabase:
    def __init__(self, runNumber, length):
        self.run = 1
        self.results_state = [[""] * length] * runNumber
        self.results_gnuString = [[""] * length] * runNumber
        self._lock = threading.Lock()

    def locked_update(self, runNumber, counter, state, gnuString):
        with self._lock:
            self.results_state[runNumber][counter] = str(state)
            self.results_gnuString[runNumber][counter] = str(gnuString)

    def set_stop(self):
        with self._lock:
            self.run = 0

    def check_run(self):
        with self._lock:
            return self.run

    def __str__(self):
     return "results_state=["+",".join(self.results_state)+"]"

    def get_all_state(self):
        return self.results_state
        
    def get_all_gnuString(self):
        return self.results_gnuString


class ResultSet:
    def __init__(self, runNumber, counter, state, gnuString):
        self.runNumber = runNumber
        self.counter = counter
        self.state = state
        self.gnuString = gnuString

    def __str__(self):
     return "state=["+",".join(self.results_state)+"]"

    def get_runNumber(self):
        return self.runNumber

    def get_counter(self):
        return self.counter
        
    def get_state(self):
        return self.state
        
    def get_gnuString(self):
        return self.gnuString

def CreateDataSet(startTime, preTime, path, file, parameters): 

    parentTrackLengthInPositions = parameters[0]
    elementsPerPositionFile = parameters[1]
    maxParentTracks = parameters[12]
    elementsPerPositionFFN = parameters[13]
    elementRatio = elementsPerPositionFile/elementsPerPositionFFN
    file_path = path + '/' + file
    
    # copying data from file to list
    trainingDataFile = open(file_path, 'r')
    trainingData = trainingDataFile.read()
    trainingDataFile.close()

    #print(trainingData[0:100])
    
    # split list containing all tracks into seperate track elements in list
    records = trainingData.split('$') # split dataset into 'records' (list of tracks)
    #print('len(records): \t' + str(len(records)))
    if len(records) > maxParentTracks:
        lenRecords = maxParentTracks
    else:
        lenRecords = len(records)
        
    print("-- trainingData.split --")
    print("number of tracks: " + str(len(records)))
    print("number of used tracks: " + str(lenRecords))

    parentDataset = np.empty([lenRecords*parentTrackLengthInPositions*elementsPerPositionFFN,], dtype=float)
    parentTrack = np.empty([parentTrackLengthInPositions*elementsPerPositionFFN,], dtype=float)
    tmpTrack = np.empty([parentTrackLengthInPositions*elementsPerPositionFile,], dtype=float)

    posTrack = 0
    while posTrack < lenRecords: 
       
        tmpTrack = np.array(records[posTrack].split(','))  # split the record by the ',' commas into individual values

        counter = 0
        while (counter*elementsPerPositionFile) < len(tmpTrack):
            parentTrack[counter * elementsPerPositionFFN + 0: counter * elementsPerPositionFFN + 5] = tmpTrack[counter * elementsPerPositionFile + 1:counter * elementsPerPositionFile + 6]
            # [unixtimestamp [s], x [m], y [m], deg [°], r [m]]
            counter = counter + 1 
            
        parentDataset[posTrack*parentTrackLengthInPositions*elementsPerPositionFFN:(posTrack+1)*parentTrackLengthInPositions*elementsPerPositionFFN] = parentTrack
        if(posTrack%(100000)) == 0:
            print('Aufbereitung Input:\t' + str(posTrack) + '/' + str(lenRecords))
        posTrack = posTrack + 1

    print('parentDataset: ' + str(int(len(parentDataset)/(parentTrackLengthInPositions*elementsPerPositionFFN))))
    preTime = statusMsg(startTime, preTime)

    #print('tmpTrack')
    #print(tmpTrack[0:12])
    #print('parentTrack')
    #print(parentTrack[0:12])
    #print('parentDataset')
    #print(parentDataset[0:12])
    #print(parentDataset)

    # free up memory        
    del trainingDataFile
    del trainingData
    del records
    del tmpTrack
    del parentTrack
    
    return preTime, parentDataset


def parentToChildTrack(startTime, preTime, parentDataset, parameters):

    parentTrackLengthInPositions = parameters[0]
    elementsPerPositionFFN = parameters[13]
    childrenTrackLengthInPositions = parameters[2]
    

    # resize form parent to child
    parentDatasetLengthInTracks = int(len(parentDataset)/(elementsPerPositionFFN*parentTrackLengthInPositions))
    #childrenDatasetLengthInTracks = int(parentTrackLengthInPositions/childrenTrackLengthInPositions)*parentDatasetLengthInTracks
    parentTrackLengthInElements = parentTrackLengthInPositions*elementsPerPositionFFN
    childrenTrackLengthPerParentTrackInElements  = int(parentTrackLengthInPositions/childrenTrackLengthInPositions)*childrenTrackLengthInPositions*elementsPerPositionFFN
    childrenTracks = np.empty([parentDatasetLengthInTracks * childrenTrackLengthPerParentTrackInElements,], dtype=float)

    #print('len(parentDataset): \t' + str(len(parentDataset)))
    #print('parentDatasetLengthInTracks: \t' + str(parentDatasetLengthInTracks))
    ##print('childrenDatasetLengthInTracks: \t' + str(childrenDatasetLengthInTracks))
    #print('parentTrackLengthInElements: \t' + str(parentTrackLengthInElements))
    #print('childrenTrackLengthPerParentTrackInElements: \t' + str(childrenTrackLengthPerParentTrackInElements))
    #print('childrenTracks: ' + str(childrenTracks))

    
    posParentTrack = 0
    while posParentTrack < (parentDatasetLengthInTracks):
        childrenTracks[(0+posParentTrack)*childrenTrackLengthPerParentTrackInElements:(1+posParentTrack)*childrenTrackLengthPerParentTrackInElements]=parentDataset[posParentTrack*parentTrackLengthInElements:posParentTrack*parentTrackLengthInElements + childrenTrackLengthPerParentTrackInElements]
        posParentTrack = posParentTrack + 1
        
    childrenTracks = childrenTracks.reshape((int(len(childrenTracks)/(elementsPerPositionFFN*childrenTrackLengthInPositions)), elementsPerPositionFFN*childrenTrackLengthInPositions))
    np.random.shuffle(childrenTracks) # to avoide that the parent track is trained
    childrenTracks = childrenTracks.reshape(parentDatasetLengthInTracks * childrenTrackLengthPerParentTrackInElements)

    print('parentToChildTrack childrenTracks: ' + str(int(len(childrenTracks)/(elementsPerPositionFFN*childrenTrackLengthInPositions))))
    preTime = statusMsg(startTime, preTime)

    #print('childrenTracks')
    #print(childrenTracks)
    #print(len(childrenTracks))
    #print(childrenTracks)

    return preTime, childrenTracks

def splitAndRescaleChildTrack(startTime, preTime, childrenTracks, parameters):

    
    
    linesInFile = 0 # count data tupels in single parentTrack

    elementsPerPositionFFN = parameters[13]
    childrenTrackLengthInPositions = parameters[2]
    scalefactor = parameters[4]
    testProcentage = parameters[5]
    precisionInDigits  = parameters[6] 

    childrenTrackLengthInElement = childrenTrackLengthInPositions*elementsPerPositionFFN
    childrenDatasetLengthInTracks = int(len(childrenTracks)/(elementsPerPositionFFN*childrenTrackLengthInPositions))
    childrenDatasetLengthInTrack100s = int(childrenDatasetLengthInTracks/100)
    
    testMeas = np.empty(childrenDatasetLengthInTrack100s * testProcentage * (childrenTrackLengthInPositions-1) * 3, dtype=float)
    testTarget = np.empty(childrenDatasetLengthInTrack100s * testProcentage * 3, dtype=float)
    trainingMeas = np.empty(childrenDatasetLengthInTrack100s * (100-testProcentage) * (childrenTrackLengthInPositions-1) * 3, dtype=float)
    trainingTarget = np.empty(childrenDatasetLengthInTrack100s * (100-testProcentage) * 3, dtype=float)

    meas = np.empty((childrenTrackLengthInPositions-1)*3, dtype=float) # for the childrenTrackLengthInPositions X,Y data pairs
    target = np.empty(1*3, dtype=float) # for the X,Y data pairs

    posChildrenTrack = 0
    posTraining = 0
    posTest = 0
    percent = 0

    posTrack = 0
    while(posTrack < len(childrenTracks)):
        childrenTracks[posTrack] = childrenTracks[posTrack] / 10000000000
        childrenTracks[posTrack + 1] = np.round((childrenTracks[posTrack + 1] / scalefactor), precisionInDigits+4)
        childrenTracks[posTrack + 2] = np.round((childrenTracks[posTrack + 2] / scalefactor), precisionInDigits+4)
        posTrack = posTrack + elementsPerPositionFFN
    

    while posChildrenTrack < (childrenDatasetLengthInTrack100s*100):

        posChildrenPosition = 0
        while posChildrenPosition < (childrenTrackLengthInPositions-1):
            meas[3*posChildrenPosition:3*posChildrenPosition+3] = childrenTracks[elementsPerPositionFFN*posChildrenPosition+posChildrenTrack*childrenTrackLengthInPositions*elementsPerPositionFFN+0:elementsPerPositionFFN*posChildrenPosition+posChildrenTrack*childrenTrackLengthInPositions*elementsPerPositionFFN+3]
            posChildrenPosition = posChildrenPosition+1 
        target[0:3] = childrenTracks[elementsPerPositionFFN*posChildrenPosition+posChildrenTrack*childrenTrackLengthInPositions*elementsPerPositionFFN+0:elementsPerPositionFFN*posChildrenPosition+posChildrenTrack*childrenTrackLengthInPositions*elementsPerPositionFFN+3]

        
        if percent < testProcentage:
            testMeas[posTest*(childrenTrackLengthInPositions-1)*3:posTest*(childrenTrackLengthInPositions-1)*3+(childrenTrackLengthInPositions-1)*3] = meas
            testTarget[posTest*3:posTest*3+3] = target
            posTest = posTest + 1
        else:
            trainingMeas[posTraining*(childrenTrackLengthInPositions-1)*3:posTraining*(childrenTrackLengthInPositions-1)*3+(childrenTrackLengthInPositions-1)*3] = meas
            trainingTarget[posTraining*3:posTraining*3+3] = target
            posTraining = posTraining + 1
            
        percent = percent + 1
        if percent == 100:
            percent = 0
        
        posChildrenTrack = posChildrenTrack+1

    print('Training Meas ' + str(int(len(trainingMeas))) + '\tTraining Target ' + str(int(len(trainingTarget))))
    print('Test Meas     ' + str(int(len(testMeas))) + '\tTest Target     ' + str(int(len(testTarget))))
    #print('trainingMeas min:' + str(min(trainingMeas)) + ' max:' + str(max(trainingMeas)))
    
    # reshape meas and target
    trainingMeas = trainingMeas.reshape(int(len(trainingMeas)/(3*(childrenTrackLengthInPositions-1))),((childrenTrackLengthInPositions-1))*3)
    trainingTarget = trainingTarget.reshape(int(len(trainingTarget)/3),3)

    testMeas = testMeas.reshape(int(len(testMeas)/(3*(childrenTrackLengthInPositions-1))),((childrenTrackLengthInPositions-1))*3)
    testTarget = testTarget.reshape(int(len(testTarget)/3),3)


    # print('Training Meas ' + str(int(len(trainingMeas)))  + ' x ' + str(int(len(trainingMeas[0]))) + ' x ' + str(int(len(trainingMeas[0][0]))) + ' Training Target ' + str(int(len(trainingTarget))) + ' x ' + str(int(len(trainingTarget[0]))) )
    # print('Test Meas     ' + str(int(len(testMeas))) + ' x ' + str(int(len(testMeas[0]))) + ' x ' + str(int(len(testMeas[0][0]))) + ' Test Target     ' + str(int(len(testTarget))) + ' x ' + str(int(len(testTarget[0]))) )
    
    print('Training Meas ' + str(int(len(trainingMeas)))  + ' x ' + str(int(len(trainingMeas[0]))) + ' Training Target ' + str(int(len(trainingTarget))) + ' x ' + str(int(len(trainingTarget[0]))) )
    print('Test Meas     ' + str(int(len(testMeas))) + ' x ' + str(int(len(testMeas[0]))) + ' Test Target     ' + str(int(len(testTarget))) + ' x ' + str(int(len(testTarget[0]))) )


   
    # rescale meas and target for KNN
    #print(testMeas[0])
    #posTrack = 0
    #while(posTrack < len(trainingMeas)):
    #    posElement = 0
    #    while(posElement<(parameters[2] - 1)):
    #        trainingMeas[posTrack][posElement] = trainingMeas[posTrack][posElement] / 10000000000
    #        trainingMeas[posTrack][posElement+1] = np.round((trainingMeas[posTrack][posElement+1] / scalefactor), precisionInDigits+4)
    #        trainingMeas[posTrack][posElement+2] = np.round((trainingMeas[posTrack][posElement+2] / scalefactor), precisionInDigits+4)
    #        posElement = posElement + 1
    #    posElement = 0
    #    while(posElement<1):
    #        trainingTarget[posTrack][posElement] = trainingTarget[posTrack][posElement] / 10000000000
    #        trainingTarget[posTrack][posElement+1] = np.round((trainingTarget[posTrack][posElement+1] / scalefactor), precisionInDigits+4)
    #        trainingTarget[posTrack][posElement+2] = np.round((trainingTarget[posTrack][posElement+2] / scalefactor), precisionInDigits+4)
    #        posElement = posElement + 1
    #    posTrack = posTrack + 1
    #posTrack = 0
    #while(posTrack < len(testMeas)):
    #    posElement = 0
    #    while(posElement<(parameters[2] - 1)):
    #        testMeas[posTrack][posElement] = testMeas[posTrack][posElement] / 10000000000
    #        testMeas[posTrack][posElement+1]= np.round((testMeas[posTrack][posElement+1] / scalefactor), precisionInDigits+4)
    #        #testMeas[posTrack][posElement+2]= (testMeas[posTrack][posElement+2] / scalefactor)
    #        #testMeas[posTrack][posElement+2]= np.round((testMeas[posTrack][posElement+2] / scalefactor), precisionInDigits+4)
    #        posElement = posElement + 1
    #    posElement = 0
    #    while(posElement<1):
    #        testTarget[posTrack][posElement] = testTarget[posTrack][posElement] / 10000000000
    #        testTarget[posTrack][posElement+1] = np.round((testTarget[posTrack][posElement+1] / scalefactor), precisionInDigits+4)
    #       testTarget[posTrack][posElement+2] = np.round((testTarget[posTrack][posElement+2] / scalefactor), precisionInDigits+4)
    #        posElement = posElement + 1
    #    posTrack = posTrack + 1

    #print('dsf')
    #print(testMeas[0])
        
    scaledTrainingMeas = trainingMeas
    scaledTrainingTarget = trainingTarget
    scaledTestMeas = testMeas
    scaledTestTarget = testTarget
    # round meas and target for KNN to avoid fake persission
    #scaledTrainingMeas = np.round(scaledTrainingMeas, precisionInDigits+4)
    #scaledTrainingTarget = np.round(scaledTrainingTarget, precisionInDigits+4)
    #scaledTestMeas = np.round(scaledTestMeas, precisionInDigits+4)
    #scaledTestTarget = np.round(scaledTestTarget, precisionInDigits+4)

    print('scaledTrainingMeas min:' + str(min(scaledTrainingMeas.flatten())) + ' max:' + str(max(scaledTrainingMeas.flatten())))
    print('scaledTrainingTarget min:' + str(min(scaledTrainingTarget.flatten())) + ' max:' + str(max(scaledTrainingTarget.flatten())))
    print('scaledTestMeas min:' + str(min(scaledTestMeas.flatten())) + ' max:' + str(max(scaledTestMeas.flatten())))
    print('scaledTestTarget min:' + str(min(scaledTestTarget.flatten())) + ' max:' + str(max(scaledTestTarget.flatten())))
    
    # print('childrenTracks: ' + str(int(len(childrenTracks)/(elementsPerPositionFFN*childrenTrackLengthInPositions))))
    preTime = statusMsg(startTime, preTime)

    
    # free up memory
    del childrenTracks
    del trainingMeas
    del trainingTarget
    del testMeas
    del testTarget
    #del parentTracks # dont remove, need for parameter iterations

    #print(scaledTestMeas[0])
    
    return preTime, scaledTrainingMeas, scaledTrainingTarget, scaledTestMeas, scaledTestTarget

def outlierGen (startTime, preTime, scaledMeas, parameters):
    elementsPerPosition = 3
    scaledMeasTrackLengthInPositions = (parameters[2] - 1) # the last position is scaledTestTarget
    scaledMeasDatesetLenghtInTrack = len(scaledMeas)
    precisionInDigits  = parameters[6] 

    scaledSigma = parameters[3] / parameters[4]
    xyOffset = np.empty(2, dtype = float)

    if scaledSigma > 0.0001:
        posTrack = 0

        randomBearings = np.empty((scaledMeasDatesetLenghtInTrack * scaledMeasTrackLengthInPositions), dtype = float)
        randomRanges = np.empty((scaledMeasDatesetLenghtInTrack * scaledMeasTrackLengthInPositions), dtype = float)

        randomBearings = np.random.uniform(0, 2 * np.pi, (scaledMeasDatesetLenghtInTrack * scaledMeasTrackLengthInPositions))
        randomRanges = np.random.normal(0, scaledSigma, (scaledMeasDatesetLenghtInTrack * scaledMeasTrackLengthInPositions))
        
        while(posTrack<scaledMeasDatesetLenghtInTrack):
            #selElement = random.randrange(0, scaledTestMeasTrackLengthInPositions)
            posElement = 0
            
            xyOffset = np.empty([scaledMeasTrackLengthInPositions, 2], dtype=float)
            
            while(posElement<scaledMeasTrackLengthInPositions):
 
                xyOffset[posElement][0] = randomRanges[posTrack * scaledMeasTrackLengthInPositions + posElement] * np.cos(randomBearings[posTrack * scaledMeasTrackLengthInPositions + posElement]) # Offset - X
                xyOffset[posElement][1] = randomRanges[posTrack * scaledMeasTrackLengthInPositions + posElement] * np.sin(randomBearings[posTrack * scaledMeasTrackLengthInPositions + posElement]) # Offset - Y




                
                #print('xyOffset: \t' + '{0:6.9f}'.format(xyOffset[0]) + ' ' + '{0:6.9f}'.format(xyOffset[1]))
                #print('xyOffset: \t' + str(xyOffset))
                #print("XES")
                #print('{0:6.9f}'.format(scaledMeas[posTrack][posElement*elementsPerPosition + 1]), '{0:6.9f}'.format(xyOffset[0]), 1)
                #print("A")
                #print(np.random.normal((scaledMeas[posTrack][posElement*elementsPerPosition + 1]), (xyOffset[0]), 1)[0])
                #print("B")
                #print(np.round((np.random.normal((scaledMeas[posTrack][posElement*elementsPerPosition + 1]), (xyOffset[0]), 1)[0]), precisionInDigits+4))
                #print('C')
                #print(scaledMeas[posTrack][posElement*elementsPerPosition + 1])
                #scaledMeas[posTrack][posElement*elementsPerPosition + 1] = np.random.normal((scaledMeas[posTrack][posElement*elementsPerPosition + 1]), (xyOffset[0]), 1)[0]
                #print(scaledMeas[posTrack][posElement*elementsPerPosition + 1])
                scaledMeas[posTrack][posElement * elementsPerPosition + 1] = scaledMeas[posTrack][posElement * elementsPerPosition + 1] + xyOffset[posElement][0]
                scaledMeas[posTrack][posElement * elementsPerPosition + 1] = np.round(scaledMeas[posTrack][posElement * elementsPerPosition + 1], precisionInDigits + 4)
                #print(scaledMeas[posTrack][posElement*elementsPerPosition + 1])
                #scaledMeas[posTrack][posElement*elementsPerPosition + 1] = np.round(np.random.normal(scaledMeas[posTrack][posElement*elementsPerPosition + 1], xyOffset[0], 1)[0], (precisionInDigits+4))
                #print(scaledMeas[posTrack][posElement*elementsPerPosition + 1])
                #print("YES")
                
                #scaledMeas[posTrack][posElement*elementsPerPosition + 2] = np.random.normal((scaledMeas[posTrack][posElement*elementsPerPosition + 2]), (xyOffset[1]), 1)[0]
                scaledMeas[posTrack][posElement * elementsPerPosition + 2] = scaledMeas[posTrack][posElement * elementsPerPosition + 2] + xyOffset[posElement][1]
                scaledMeas[posTrack][posElement * elementsPerPosition + 2] = np.round(scaledMeas[posTrack][posElement * elementsPerPosition + 2], precisionInDigits + 4)
                #print("ZES")
                # scaledMeas[posTrack][posElement*elementsPerPosition + 2] = np.round((float(scaledMeas[posTrack][posElement*elementsPerPosition + 2]) + (random.uniform(-  xyOffset[1],  xyOffset[1]))), precisionInDigits+4)
                #print(scaledMeas[posTrack][posElement*elementsPerPosition + 1])


                
                posElement = posElement + 1
            #print(scaledMeas[posTrack])
            #print('---')
            posTrack = posTrack + 1

    preTime = statusMsg(startTime, preTime)

    #print('ls')
    #print(scaledMeas[0:3])
    

    return preTime, scaledMeas

def FFN(startTime, preTime, scaledTrainingMeas, scaledTrainingTarget, scaledTestMeas, scaledTestTarget, parameters):
    ##Var

    
    childrenTrackLengthInPositions = parameters[2]
    scalefactor = parameters[4]
    precisionInDigits  = parameters[6] 
    unitS = parameters[7]
    batchS = parameters[8]
    epochS = parameters[9]
    layerS = parameters[10]
    offset = parameters[11]
    activationFunction = parameters[14]
    optimiZer = parameters[16]
    learningRate = parameters[17]
    dropoutRate = parameters[18]
    
    ### Mnist
    # https://pyimagesearch.com/2021/05/06/implementing-feedforward-neural-networks-with-keras-and-tensorflow/


    # Reshape so that Target Time is an input value (TestX/TrainX) 
    trainX = np.empty([len(scaledTrainingMeas), (parameters[2] - 1) * 3], dtype=float)
    trainY = np.empty([len(scaledTrainingMeas), 1 * 3], dtype=float)
    testX = np.empty([len(scaledTestMeas), (parameters[2] - 1) * 3], dtype=float)
    testY = np.empty([len(scaledTestMeas), 1 * 3], dtype=float)


    trainX = scaledTrainingMeas
    trainY = scaledTrainingTarget
    testX = scaledTestMeas
    testY = scaledTestTarget

    # Reshape so that Target Time is an input value (TestX/TrainX) 
    #trainX = np.empty([len(scaledTrainingMeas), (parameters[2] - 1) * 3 + 1], dtype=float)
    #trainY = np.empty([len(scaledTrainingMeas), 1 * 2], dtype=float)

    #print(trainX.shape)
    #print(scaledTrainingMeas.shape)
    #print(trainY.shape)
    #print(scaledTrainingTarget.shape)
    
    #testX = np.empty([len(scaledTestMeas), (parameters[2] - 1) * 3 + 1], dtype=float)
    #testY = np.empty([len(scaledTestMeas), 1 * 2], dtype=float)


    
    #print(scaledTrainingMeas[0], scaledTrainingTarget[0][0])
    #print(np.append(scaledTrainingMeas[0], scaledTrainingTarget[0][0]))
    #posTrack = 0
    #while(posTrack < len(scaledTrainingMeas)):
    #    trainX[posTrack] = np.append(scaledTrainingMeas[posTrack], scaledTrainingTarget[posTrack][0])
    #    trainY[posTrack] = scaledTrainingTarget[posTrack][1:3] #np.round(scaledTrainingTarget, precisionInDigits)
    #    posTrack = posTrack + 1
    #    print(trainX)

    #while(posTrack < len(scaledTestMeas)):
    #    testX[posTrack] = np.append(scaledTestMeas[posTrack], scaledTestTarget[posTrack][0])
    #    testY[posTrack] = scaledTestTarget[posTrack][1:3] #np.round(scaledTestTarget, precisionInDigits)
    #    posTrack = posTrack + 1
    
    #print('trainX \t' + str(int(len(trainX))) + '\t trainY \t\t' + str(int(len(trainY))))
    #print('testX  \t' + str(int(len(testX))) + '\t testY  \t\t' + str(int(len(testY))))
    #print('Batchs \t' + str(batchS) + '\t Epochs  \t\t' + str(epochS))
    #print('Units  \t' + str(unitS) + '\t Layers  \t\t' + str(layerS))
    #print('Offset \t' + str(offset) + '\t precisionInDigits \t' + str(precisionInDigits))
    #print("----")
    #print(scaledTrainingMeas[189])
    #print(scaledTrainingTarget[189])
    
    #model = keras.models.Sequential()
    model = Sequential()

    counter = 0
    while counter < layerS:
        model.add(Dropout(dropoutRate))
        #configuring hidden layers
        model.add(Dense(units = unitS,
                        input_dim=(len(trainX[0])), # size of the trainX + time trainY
#                         kernel_initializer = 'normal',
                        activation = activationFunction, 
                        kernel_initializer='random_normal',
                        bias_initializer='zeros'
                        ))
        counter = counter + 1

    
        
    # create more hidden layers if needed
    # model.add(Dense(units = unitS, kernel_initializer = 'normal', activation = 'sigmoid'))

    # definition of the output layer (two output values)
    #print("[INFO] model add...")
    model.add(Dense(units = 3, kernel_initializer = 'random_normal'))        # !=time, x, y

    #model.summary() # Summary of the network strukture
    #print("[INFO] learning rate")
    learning_rate = tf.Variable(learningRate, trainable=False)
    # train the model using adam
    print("[INFO] training network... compile")
    model.compile(loss='mean_squared_error', optimizer=optimiZer,  metrics=["accuracy"])
    print("[INFO] training network... fit")
    modelFit = model.fit(trainX, trainY, batch_size = batchS, epochs=epochS, verbose=0, shuffle = True) #verbose=2 detaild verbose=0 off
    #model_path = path[:len(path)-14] + '/AIS-KNN-Models/' + file[:len(file)-4] + '_model.h5'
    #if os.path.isfile(model_path) ist False:
    #    model.save(model_path)
    
    
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=batchS, verbose=0)
    # Hier fehlt eine Formel, zum um den Mittlerenqudaratischen Fehler numpy.sqrt((testY[1][1]-prdict[x][1])**2+(testY[1][2]-prdict[x][2])**2)
    # Überlegen ob die Ausgabe so passend ist oder ob da nicht noch etwas verbessert werden kann.
    # print('Y')
    # print(testY[:16])
    # print('pred')
    # print(predictions[:16])
    
    
    gnuPosition = [0] * 3
    gnuString = [0]  * len(testY)
    posPosition = 0
    

    while posPosition < (len(testY)):
        gnuPosition[1] = '0, 0,' + ', '.join(map(str, testY[posPosition]))
        gnuPosition[2] = '0, 0,' + ', '.join(map(str, predictions[posPosition]))
        posPosition = posPosition + 1
    gnuString = '\n'.join(map(str, gnuPosition))
    gnuString = gnuString + '\n'
    #print(gnuString)
          
    sumErrorDist = 0
    position = 0
    deltaDist = np.empty(len(testY), dtype=float)
    statistic = [0] * 8
    while position < len(testY):
        deltaDist[position] =  np.sqrt((abs(testY[position][1])-abs(predictions[position][1]))**2+(abs(testY[position][2])-abs(predictions[position][2]))**2)
        position = position + 1
    # differencValToPred = abs(testY) - abs(predictions)
    statistic[0] = len(trainX)*len(trainX[0]) # Number of Training X Data
    statistic[1] = len(trainY)*len(trainY[0]) # Number of Training Y Data
    statistic[2] = len(testX)*len(testX[0]) # Number of Test X Data
    statistic[3] = len(testY)*len(testY[0]) # Number of Test Y Data
    statistic[4] = round(np.average(deltaDist)*scalefactor, precisionInDigits)
    statistic[5] = round(np.std(deltaDist)*scalefactor, precisionInDigits)
    statistic[6] = round(np.max(deltaDist)*scalefactor, precisionInDigits)
    statistic[7] = round(np.min(deltaDist)*scalefactor, precisionInDigits)
    
    print('Mittelwert Abweichung: ' + '{0:6.1f}'.format(statistic[4]) + ' m \t' + 'Standardabweichung Abweichung: ' + '{0:6.1f}'.format(statistic[5]) + ' m')
    #print('Standardabweichung Abweichung: ' + '{0:6.1f}'.format(statistic[3]) + ' m')
    #print('Maximale Abweichung: ' + '{0:6.1f}'.format(statistic[4]) + ' m')
    #print('Minimale Abweichung: ' + '{0:6.1f}'.format(statistic[5]) + ' m')
    

    # # calculation of the differenz between actual and predicted value
    # differencValToPredRMS = []
    # counter = 0
    # for elements in testY:
    #     differencValToPredRMS = differencValToPredRMS + math.sqrt(abs()-abs())
        
    #     differencevalT = differencevalT + [[math.sqrt(abs((predictedvalT[counter][1]*predictedvalT[counter][1])+(targetval_testCTT[counter][1]*targetval_testCTT[counter][1])-(2*predictedvalT[counter][1]*targetval_testCTT[counter][1]*math.cos(math.radians(abs(predictedvalT[counter][0]-targetval_testCTT[counter][0]))))))]]
        
    #     if targetval_testCTT[counter][0] > predictedvalT[counter][0]:
    #         differenceval[counter] = [abs(((differenceval[counter][0]/360)*2*pi*predictedvalT[counter][1])), abs(differenceval[counter][1])]
    #     else:
    #         differenceval[counter] = [abs(((differenceval[counter][0]/360)*2*pi*targetval_testCTT[counter][1])), abs(differenceval[counter][1])]
        
    #     # calculation of accuracy and loss of the current prediction
    #     # Mean absolute error = MAPE
    #     # Mean squared loss = MSL
    #     maec += [[abs(differencevalT[counter][0]/6000)]]
    #     counter += 1
    
    # # calculation of accuracy and loss of the iteration

    # # Mean absolute error = MAE\n",
    # # Mean squared loss = MSL\
    # print(np.mean(maec, dtype = np.float64))
    # mae = 100 - 100*np.mean(maec, dtype = np.float64)
    # msl = np.mean(pow(targetval_testCT - predictedval, 2))
    # print(mae)
    # # save the general result for configuration in results
    # configuration_results += [[batch, epoch, layer, unit, mae, msl]]

    
    # print("-- classification_report /sqrt(delatx**2 + deltay**2) --")
    # # print(root_mean_squared_error((testY[:,0]*10**precisionInDigits).astype("int"), (predictions[:,0]*10**precisionInDigits).astype("int")))
    # print("-- classification_report X axis --")
    # print(classification_report((testY[:,0]*10**precisionInDigits).astype("int"), (predictions[:,0]*10**precisionInDigits).astype("int")))
    # print("-- classification_report Y axis --")
    # print(classification_report((testY[:,1]*10**precisionInDigits).astype("int"), (predictions[:,1]*10**precisionInDigits).astype("int")))
    # #print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))
    # #print(classification_report((testY[:,0]*100).astype("int"), (predictions[:,0]*100).astype("int"), target_names=[str(x) for x in np.unique((testY[:,0]*100).astype("int"))]))
    # print("-- classification_report END --")
    
    ## plot the training loss ##and accuracy
    ##print('History:' + str(modelFit.history))
    #plt.style.use("ggplot")
    #plt.figure()
    #plt.plot(np.arange(1,1+epochS), modelFit.history["loss"], label="train_loss")
    ##plt.plot(np.arange(0,epochS), modelFit.history["val_loss"], label="val_loss")
    #plt.plot(np.arange(0,epochS), modelFit.history["accuracy"], label="train_acc")
    ##plt.plot(np.arange(0,epochS), modelFit.history["val_accuracy"], label="val_acc")
    #plt.title("Training Loss and Accuracy")
    #plt.xlabel("Epoch #")
    #plt.ylabel("Loss/Accuracy")
    #plt.legend()
    ##plt.savefig(args["output"])

    

    
    preTime = statusMsg(startTime, preTime)

    del model
    #keras.backend.clear_session()
    #keras.backend.clear_session()

    gc.collect()

    #preTime = statusMsg(startTime, preTime)

    return statistic, gnuString

def statusMsg(startTime, preTime):
    print('Laufzeit insg: \t' + '{0:6.3f}'.format((time.time() - startTime)) + '\tLaufzeit Fkt: \t' + '{0:6.3f}'.format((time.time() - preTime)) + '\tRAM in MiB: ' + '{0:.1f}'.format(psutil.Process().memory_info()[1] / float(2**20)))
    preTime = time.time()
    return(preTime)
#----------aisdkRAWtest_15_kkn
#mainKNN_T('/home/sebastian/Dokumente/AIS-KNN-Files', 'aisdk-2023-11-08-7xs_1_kkn.csv')
#mainKNN_T('/home/sebastian/Dokumente/AIS-KNN-Files', 'aisdk-2023-11-08-6xs_1_kkn.csv')
#mainKNN_T('/home/sebastian/Documents/AIS-KNN-Files', 'aisdk-2023-11-08-6xs_5_kkn.csv')
#mainKNN_T('/home/sebastian/Documents/AIS-KNN-Files', 'aisdk-2023-11-08-xxxxs_1_kkn.csv')
#mainKNN_T('/home/sebastian/Dokumente/AIS-KNN-Files', 'aisdk-2023-11-08-5xs_1_kkn.csv')
#mainKNN_T('/home/sebastian/Dokumente/AIS-KNN-Files', 'aisdk-2023-11-08-xxxxs_1_kkn.csv')
mainKNN_T('/home/sebastian/Dokumente/AIS-KNN-Files', 'aisdk-2023-11-08-xxxs_1_kkn.csv')
#mainKNN_T('/home/sebastian/Dokumente/AIS-KNN-Files', 'aisdk-2023-11-08-xxs_1_kkn.csv')
#mainKNN_T('/home/sebastian/Dokumente/AIS-KNN-Files', 'aisdk-2023-11-08-xs_1_kkn.csv')
#mainKNN_T('/home/sebastian/Dokumente/AIS-KNN-Files', 'aisdk-2023-11-08-xs_4_kkn.csv')
#mainKNN_T('/home/sebastian/Dokumente/AIS-KNN-Files', 'aisdk-2023-11-08-xs_5_kkn.csv')
#mainKNN_T('/home/sebastian/Dokumente/AIS-KNN-Files', 'aisdk-2023-11-08-s_1_kkn.csv')
#mainKNN_T('/home/sebastian/Dokumente/AIS-KNN-Files', 'aisdk-2023-11-08-s_2_kkn.csv')
#mainKNN_T('/home/sebastian/Dokumente/AIS-KNN-Files', 'aisdk-2023-11-08_1_kkn.csv')




# mainKNN_T("C:\\Dokumente\\Studium\\Jade HS\\MA_ET\\AIS_MA\\aisdk-2023-11-11_6x_kkn.csv")
# mainKNN_T("C:\\Dokumente\\Studium\\Jade HS\\MA_ET\\AIS_MA\\aisdk-2023-11-11_kkn.csv")
# mainKNN_T("C:\\Dokumente\\Studium\\Jade HS\\MA_ET\\AIS_MA\\aisdk-2023-11-11_filtered_kkn.csv")
# mainKNN_T("C:\\Dokumente\\Studium\\Jade HS\\MA_ET\\AIS_MA\\aisdk-2023-11-11_266288000_kkn.csv")
# mainKNN_T("C:\\Dokumente\\Studium\\Jade HS\\MA_ET\\AIS_MA\\aisdk-2023-11-11_266288000_s_kkn.csv")
# mainKNN_T("C:\\Dokumente\\Studium\\Jade HS\\MA_ET\\AIS_MA", "aisdk-2023-11-11_266288000_xs_kkn.csv")

# ToDo
# aufteilen von CreateDataSet in Daten als record einlesen, in ParentTrack[100] zerlegen
# neue Funktion ParentTrack[100] in resizedTrack[15] aufteilen und dann in meas und target aufteilen
# resizedTrack[15] zu scaledTrack [15]