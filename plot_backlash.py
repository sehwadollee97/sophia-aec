# https://learnopencv.com/camera-calibration-using-opencv/
# https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/

from turtle import left, position
import cv2
from torch import t
import yaml.dumper
import numpy as np
import os
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import math

import numpy.matlib as numat

from sophia import Sophia
from AECControl import AECControl

eye_type='left'
method='1'

if method=='2' or method=='3':

    delta_motor=pd.read_csv('linear_factors/{}_eye/{}_backlash_deltaMotor_method{}.csv'.format(eye_type,eye_type, method), header=None)
    position_actual=pd.read_csv('linear_factors/{}_eye/{}_backlash_actualdeltaMotor_method{}.csv'.format(eye_type,eye_type, method),header=None)

    delta_motor=np.array(delta_motor)
    position_actual=np.array(position_actual)

    dim=int(math.sqrt(position_actual.shape[0]))


    P_command=np.reshape(delta_motor[:, 0], (dim, dim))
    T_command=np.reshape(delta_motor[:, 1],  (dim, dim))

    pan_actual=position_actual[:, 0].reshape((dim, dim))
    tilt_actual=position_actual[:, 1].reshape((dim, dim))
    ref_pan=pan_actual[3,3]
    ref_tilt=tilt_actual[3,3]

    delta_pan=pan_actual-numat.repmat(ref_pan, dim, dim)
    delta_tilt=tilt_actual-numat.repmat(ref_tilt, dim, dim)

    mse=np.mean((P_command-delta_pan)**2+(T_command-delta_tilt)**2)
    print('mse: ', mse)

    fig=plt.figure()

    plt1=fig.add_subplot(121)
    plt2=fig.add_subplot(122)

    plt1.scatter(P_command, T_command)
    plt1.set_title('Commanded delta_motor')
    plt1.set_xlabel('delta_pan')
    plt1.set_ylabel('delta_tilt')

    plt2.scatter(delta_pan, delta_tilt)
    plt2.set_title('actual delta motor')
    plt2.set_xlabel('delta pan')
    plt2.set_ylabel('delta tilt')

    plt.savefig('linear_factors/{}_eye/{}_backlash_method{}_mse{}.png'.format(eye_type,eye_type, method, mse))

    plt.show()

    plt.figure()
    plt.scatter(P_command, T_command, cmap='green')
    plt.scatter(delta_pan,delta_tilt, cmap='red')
    plt.savefig('linear_factors/{}_eye/{}_backlash_method{}mse{}_all.png'.format(eye_type,eye_type, method, mse))
    plt.show()
else:
    position_actual=pd.read_csv('linear_factors/{}_eye/{}_backlash_actualdeltatilt_method1.csv'.format(eye_type,eye_type),header=None)
    position_actual=np.array(position_actual)
    print(position_actual.shape)
    p=0
    t=20

    p_cmd=np.repeat(np.array([p]),12)
    t_cmd=np.repeat(np.array([t]), 12)

    pan_actual=position_actual[:, 0]
    tilt_actual=position_actual[:, 1]
    dim=pan_actual.shape[0]
    delta_pan=[]
    delta_tilt=[]

    for i in range(dim-1):
        delta=pan_actual[i+1]-pan_actual[i]
        delta_pan.append(delta)
    
    for i in range(dim-1):
        delta=tilt_actual[i+1]-tilt_actual[i]
        delta_tilt.append(delta)

    delta_pan=np.array([delta_pan])
    print(delta_pan)
    delta_tilt=np.array([delta_tilt])
    print(delta_tilt)
    mse=np.mean((delta_pan-np.repeat(p, dim-1))**2+ (delta_tilt-np.repeat(t, dim-1))**2)
    print(mse)
    plt.figure()
    

    plt.scatter(delta_pan,delta_tilt, color='red')
    plt.scatter(p_cmd,t_cmd, color='green')
    plt.show()

    
    fig=plt.figure()
    plt1=fig.add_subplot(121)
    plt2=fig.add_subplot(122)

    plt1.plot(np.arange(dim-1), np.reshape(delta_pan, (dim-1, 1)), color='red')
    plt1.plot(np.arange(dim-1), np.repeat(p, dim-1), color='green')

    plt2.plot(np.arange(dim-1), np.reshape(delta_tilt, (dim-1, 1)),color='red')
    plt2.plot(np.arange(dim-1), np.repeat(t, dim-1), color='green')
    plt.savefig('linear_factors/{}_eye/{}_backlash_method1tilt{}_mse{}.png'.format(eye_type,eye_type, method, mse))
    plt.show()


    #dim=int(math.sqrt(position_actual.shape[0]))



