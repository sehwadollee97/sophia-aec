# https://learnopencv.com/camera-calibration-using-opencv/
# https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/

from turtle import left
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


from sophia import Sophia
from AECControl import AECControl

# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


# Extracting path of individual image stored in a given directory
# dir=os.getcwd()
# print('dir', dir)
# images = cv2.imread('dir{}/left350.png'.format(dir))
left = Sophia(14)
right = Sophia(15)
updown = Sophia(16)



# ####################################################################################
# #calibration

# min_left = left.calibration(0)
# max_left = left.calibration(1)
# left.move((min_left+max_left)/2)
#left.write()



# min_right = right.calibration(0)
# max_right = right.calibration(1)
# right.move((min_right+max_right)/2)
#right.write()


# min_updown = updown.calibration(0)
# max_updown = updown.calibration(1)
# updown.move((min_updown+max_updown)/2)
#updown.write()

# print('min left: {0} & max left:{1}  '.format(min_left,max_left) )
# # print('min right: {0} & max right: {1}'.format( min_right,max_right))
# print('min updown: {0}, max_updown: {1}'.format(min_updown,max_updown))

# #obain the current position of the left and right
# left_currentpos=left.get_currentPos()
# right_currentpos=right.get_currentPos()
# updown_currentpos=updown.get_currentPos()

#print('current left: {0}, current right: {1}, current updown: {2}'.format( left_currentpos, right_currentpos, updown_currentpos))
####################################
position_left_prev=left.get_currentPos()
pan_position_prev=right.get_currentPos()
position_updown_prev=updown.get_currentPos()

print('pan_position_prev', pan_position_prev)
print('position_updown_prev', position_updown_prev)
print('position_left_prev', position_left_prev)
#movement
# pan=[-40, -20, 0, 20, 40]
# tilt=[-40, -20, 0, 20, 40]
# print(pan[1])
eye_type='left'
delta_pan=[0] 
delta_tilt=[20]
left_images=[]
left_image_corners=[]
# with open('linear_factors/{}_eye/{}_backlash_deltaMotor.csv'.format(eye_type,eye_type), 'r+') as f:
#     f.truncate(0)

# with open('linear_factors/{}_eye/{}_backlash_actualdeltaMotor.csv'.format(eye_type,eye_type), 'r+') as f:
#     f.truncate(0)
with open('linear_factors/{}_eye/{}_backlash_deltatilt_method1.csv'.format(eye_type,eye_type), 'ab') as f:
    f.truncate(0)
with open('linear_factors/{}_eye/{}_backlash_actualdeltatilt_method1.csv'.format(eye_type,eye_type), 'ab') as f:
    f.truncate(0)
min_r, max_r, mid_r=right.get_limit()
right.move(min_r)
min_u, max_u, mid_u=updown.get_limit()
updown.move(mid_u)
# print(u, u1)
#updown.moveby(t)


#left.moveby(p)

#calculate how much iterations it requires



pan_position=left.get_currentPos()
tilt_position=updown.get_currentPos()

p=0
t=20

if eye_type=='left':
    min_l, max_l, mid_l=left.get_limit()
    #it_pan=int((max_l-min_l)/p)
    it_tilt=int((max_u-min_u)/t)
    print(it_tilt)
#print(u, u1)
    left.move(min_l)
    
    
    for i in range(it_tilt):

        left.moveby(p)
        updown.moveby(t)


        #move sophia's eyes to the desired pan and tilt positions
        pan_position=left.get_currentPos()
        tilt_position=updown.get_currentPos()

        actual_motor=np.array([pan_position, tilt_position])
        actual_motor=actual_motor.reshape((1,2))
        with open('linear_factors/{}_eye/{}_backlash_actualdeltatilt_method1.csv'.format(eye_type,eye_type), 'ab') as f:
            np.savetxt(f, actual_motor, delimiter=',')

        
     
    
if eye_type=='right':

    pan_position=right.get_currentPos()
    tilt_position=updown.get_currentPos()

    actual_motor=np.array([pan_position, tilt_position])
    with open('linear_factors/{}_eye/{}_backlash_actualdeltaMotor.csv'.format(eye_type,eye_type), 'ab') as f:
        np.savetxt(f, delta_motor, delimiter=',')


    #optical center coordinates
    c_x=345
    c_y=252

    #move sophia's eyes to the desired pan and tilt positions
    _, _, mid_r=right.get_limit()
    right.move(mid_r)

    u, u1, mid_u=updown.get_limit()
    print(u, u1)
    updown.moveby(t)

    _, _, mid_l=left.get_limit()
    #print(u, u1)
    right.moveby(p)


    new_pan_position=right.get_currentPos()
    new_tilt_position=updown.get_currentPos()
    print('new_pan_position', new_pan_position)
    print('new_position_updown', new_tilt_position)


    new_actual_motor=np.array([new_pan_position, new_tilt_position])
    with open('linear_factors/{}_eye/{}_backlash_actualdeltaMotor3.csv'.format(eye_type,eye_type), 'ab') as f:
        np.savetxt(f, new_actual_motor, delimiter=',')


    