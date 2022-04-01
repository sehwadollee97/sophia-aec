#figures
import matplotlib.pyplot as plt
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
import matplotlib.patches as patches
import pandas as pd


from sophia import Sophia
from AECControl import AECControl
from Agent import Agent

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

agent=Agent(10)
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
delta_pan=[ 0] 
delta_tilt=[0]
left_images=[]
left_image_corners=[]


with open('linear_factors/{}_eye/{}_deltaMotor.csv'.format(eye_type,eye_type), 'ab') as f:
    f.truncate(0)


with open('linear_factors/{}_eye/{}_panPos.csv'.format(eye_type,eye_type),'r+') as f:
    f.truncate(0)

with open('linear_factors/{}_eye/{}_tiltPos.csv'.format(eye_type,eye_type),'r+') as f:
    f.truncate(0)
with open('linear_factors/{}_eye/{}_Coord_data.csv'.format(eye_type,eye_type), 'r+') as f:
    f.truncate(0)


for i in range(len(delta_pan)):
    for j in range(len(delta_tilt)):
        print('delta_pan, tilt',delta_pan[i],delta_tilt[j] )
        #print('delta_tilt: \n',delta_tilt[j] )
        print('i,j: ', i, j)
        ################ MOTOR COMMAND#################
        p=30+delta_pan[i]
        t=-180+delta_tilt[j]
        ################ MOTOR COMMAND#################
        delta_motor=np.array([delta_pan[i], delta_tilt[j]])
        delta_motor=delta_motor.reshape((1,2))

        with open('linear_factors/{}_eye/{}_deltaMotor.csv'.format(eye_type,eye_type), 'ab') as f:
            np.savetxt(f, delta_motor, delimiter=',')

        if eye_type=='left': 

            #pan = int((self.img_dim_x/2-self.t_x)*-2.770507588259701892e-01+(self.img_dim_y/2-self.t_y)* -6.221764284096930470e-02) # how much to move horizontally in motor units
            #tilt =int ((self.img_dim_x/2-self.t_x)*8.373739090644946679e-04 + (self.img_dim_y/2-self.t_y)*7.667757295016363051e-01) # how much to move vertically in motor units
            mat_A=np.array([[-2.770507588259701892e-01, -6.221764284096930470e-02],[8.373739090644946679e-04,7.667757295016363051e-01]])

            delta_motor=np.array([delta_pan[i],delta_tilt[j]] )
            delta_pixel=np.dot(np.linalg.inv(mat_A),delta_motor)
            patch_size=150
            w=int(patch_size/2)
            left_cap=cv2.VideoCapture(0)

            _, left_img_bs = left_cap.read() 
            # print(np.shape(left_img))
            left_cap.release()

            #optical center coordinates
            c_x=345
            c_y=252

            #move sophia's eyes to the desired pan and tilt positions
            _, _, mid_r=right.get_limit()
            right.move(mid_r)

            u, u1, mid_u=updown.get_limit()
            print(u, u1)
            updown.move(mid_u+t)

            _, _, mid_l=left.get_limit()
            #print(u, u1)
            left.move(mid_l+p)

            pan_position=left.get_currentPos()
            tilt_position=updown.get_currentPos()
        
            print('pan_position', pan_position)
            print('position_updown', tilt_position)
            
            pan_position=np.array([pan_position])
            tilt_position=np.array([tilt_position])
            left_cap = cv2.VideoCapture(0)

            _, left_img_as = left_cap.read() 
            # print(np.shape(left_img))
            left_cap.release()
            hei=left_img_as.shape[0]
            wid=left_img_as.shape[1]
            #bs patch

            bs_patch=left_img_as[int(c_x+delta_pixel[0]-w):int(c_x+delta_pixel[0]+w), c_y-w: c_y+w]
            as_patch=left_img_as[c_x-w:c_x+w, c_y-w: c_y+w]

            fig=plt.figure()

            #add subplots
            plt1=fig.add_subplot(221)
            plt2=fig.add_subplot(222)
            plt3=fig.add_subplot(223)
            plt4=fig.add_subplot(224)
            #bs_patch=left_img_bs[]
            #im_BS = self.imagedata[self.y_before:self.y_after, self.x_before:self.x_after]
            fig=plt.figure()

            plt1.plot([0, wid], [c_y, c_y])
            plt1.plot([c_x, c_x], [0, hei])
            plt1.scatter(delta_pixel[0], c_y)
            plt1.imshow(left_img_bs)
            

            plt2.plot([0, wid], [c_y, c_y])
            plt2.plot([c_x, c_x], [0, hei])
            plt2.scatter(delta_pixel[0], delta_pixel[1])
            plt2.add_patch(patches.Rectangle((delta_pixel[0]-w, delta_pixel[1]-w), patch_size, patch_size, fill=False))
            plt2.imshow(left_img_as)

            plt3.imshow(bs_patch)
            plt4.imshow(as_patch)
            plt.savefig('/home/fyp/Downloads/SCServo_Python_200831/SCServo_Python/saccade_test/{}/image_deltap{}_deltat{}.png'.format(eye_type, delta_pan[i], delta_tilt[j]))
            plt.show()

                        
            
        if eye_type=='right':


            #optical center coordinates
            c_x=320
            c_y=246

            #move sophia's eyes to the desired pan and tilt positions
            _, _, mid_r=right.get_limit()
            right.move(mid_r+p)

            u, u1, mid_u=updown.get_limit()
            print(u, u1)
            updown.move(mid_u+t)

            _, _, mid_l=left.get_limit()
            #print(u, u1)
            left.move(mid_l)


            pan_position=right.get_currentPos()
            tilt_position=updown.get_currentPos()
            #position_left=left.get_currentPos()

            #save the pan and tilt position in the file
            
            #tilt_pos=np.append(tilt_pos, position_updown)
            #print(tilt_pos.shape)\
            print('pan_position', pan_position)
            print('position_updown', tilt_position)
            
            pan_position=np.array([pan_position])
            tilt_position=np.array([tilt_position])

       
            
            right_cap = cv2.VideoCapture(2)
            #left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            #left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            _, right_img = right_cap.read() 
            # print(np.shape(right_img))
            right_cap.release()
            #right_img=cv2.resize(right_img, (320, 240))
            hei=right_img.shape[0]
            wid=right_img.shape[1]
            plt.figure()
            plt.plot([0,wid ], [c_y, c_y])
            plt.plot([c_x, c_x ], [0, hei])
            #mpimg.imsave('linear_factors/{}_eye/images/{}_{}.png'.format(eye_type, delta_pan[i] ,delta_tilt[j]),right_img)
            
            plt.imshow(right_img)
            plt.savefig('linear_factors/{}_eye/images/corners/{}_{}.png'.format(eye_type, delta_pan[i] ,delta_tilt[j]))
            plt.show()
            
            print('i,j: ', i, j)
            print('delta_pan, tilt',delta_pan[i],delta_tilt[j] )

            with open('linear_factors/{}_eye/{}_panPos.csv'.format(eye_type,eye_type), 'ab') as f:
                np.savetxt(f, pan_position, delimiter=',')
            with open('linear_factors/{}_eye/{}_tiltPos.csv'.format(eye_type,eye_type), 'ab') as f:
                np.savetxt(f,tilt_position, delimiter=',')

            
            #images=glob.glob('./*L?.png')
            for fname in right_img:
                img=right_img
                
                
                
                #img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Find the chess board corners
                # If desired number of corners are found in the image then ret = true
                ret, corners = cv2.findChessboardCorners(
                    gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

                """
                If desired number of corner are detected,
                we refine the pixel coordinates and display 
                them on the images of checker board
                """
                print('ret:  ', ret)
                
                if ret == True:
                    

                    
                    #plt.savefig('right_img.png')
                    objpoints.append(objp)
                    # refining pixel coordinates for given 2d points.
                    corners2 = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria)
                    corners2=np.reshape(corners2, (54, 2))
                    image = cv2.drawChessboardCorners(right_img,
                                          CHECKERBOARD,
                                          corners2, ret)
                    print(corners2)
                    print(corners2.shape)
                    print(corners2[27, 0], corners2[27, 1])
 
                    plt.figure()
                    
                    #plt.scatter(corners2[27, 0], corners2[27, 1],marker='>', cmap='purple', s=100 )
                    plt.imshow(image)
                    plt.savefig('linear_factors/{}_eye/images/{}_{}_nc.png'.format(eye_type, delta_pan[i] ,delta_tilt[j]))
                    
                    plt.show()
                    #mpimg.imsave('linear_factors/{}_eye/images/{}_{}_nc.png'.format(eye_type,delta_pan[i] ,delta_tilt[j]),image)
                    
                    
                    with open('linear_factors/{}_eye/{}_Coord_data.csv'.format(eye_type,eye_type), 'ab') as f:
                        np.savetxt(f, corners2, delimiter=',')
                    with open('linear_factors/{}_eye/{}_Coord_data27.csv'.format(eye_type,eye_type), 'ab') as f:
                        np.savetxt(f, corners2[27,:], delimiter=',')
                   
                    break
        
#####################################
                    #imgpoints.append(corners2)
                # else:
                #     with open('linear_factors/{}_eye/{}_Coord_data.csv'.format(eye_type,eye_type), 'ab') as f:
                #         np.savetxt(f, imgpoints, delimiter=',')
                #     quit()
            

        
        
        #saving the pan and tilt positions
        
        
        
        #saving 

    #         # writing the corners to the txt file
    #         #print('range(0, len(right_img))',  len(right_img))
            

    #         # with open('corners_updown{}_right_pan{}.csv'.format(position_updown, pan_position), 'w') as f:
    #         #     for line in corners:
    #         #         line=str(line)
    #         #         f.writelines(line) # writing tuples
    #         #         f.writelines('\n') # writing tuples
    #         with open('linear_factors/{}_eye/{}_pan{}_tilt{}.csv'.format(eye_type, eye_type,pan_position, position_updown), 'a') as f:
    #             for line in corners2:
    #                 #f.writelines('\n') # writing tuples
    #                 np.savetxt(f, line, delimiter=',')
    #                 #f.writelines('\n') # writing tuples



    #         print('corners:  ', corners2)
    #         print('type of corners:  ', type(corners2))
    #         print('size of corners:  ', (np.shape(corners2)))
    #         #print('corner2 {0}th row, 1st column, {1}th element'.format(1,1), corners2[0][0][0])
    #         img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    #         cv2.imshow('linear_factors/{}_eye/{}_pan{}_tilt{}'.format(eye_type, eye_type,pan_position, position_updown),img)
            
    #         file_dir='/home/fyp/Downloads/SCServo_Python_200821/right eye calibration'
            
    #         mpimg.imsave('linear_factors/{}_eye/{}_pan{}_tilt{}.png'.format(eye_type, eye_type,pan_position, position_updown), img)
        
    #     #cv2.imshow('img',img)
    #     #cv2.waitKey(0)
    #         ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    #         objpoints, imgpoints, gray.shape[::-1], None, None)
    #         print("Camera matrix : \n")
    #         print(mtx)
    #         print("dist : \n")
    #         print(dist)
    #         print("rvecs : \n")
    #         print(rvecs)
    #         print("tvecs : \n")
    #         print(tvecs)
    #         with open('linear_factors/{}_eye/camera_mat_{}_pan{}_tilt{}.csv'.format(eye_type, eye_type,pan_position, position_updown), 'a') as f:
                    
    #                     #f.writelines('\n') # writing tuples
    #                 np.savetxt(f, mtx, delimiter=',')
    #     # with open('linear_factors/{}_eye/rvec_mat_{}_pan{}_tilt{}.csv'.format('right', 'right',pan_position, position_updown), 'a') as f:
                
    #     #             #f.writelines('\n') # writing tuples
    #     #         np.savetxt(f, rvecs, delimiter=',')
    #         with open('linear_factors/{}_eye/tvec_mat_{}_pan{}_tilt{}.csv'.format(eye_type, eye_type,pan_position, position_updown), 'a') as f:
    #                 for line in tvecs:
    #                     #f.writelines('\n') # writing tuples
    #                     np.savetxt(f, line, delimiter=',')
    #         with open('linear_factors/{}_eye/rvec_{}_pan{}_tilt{}.csv'.format(eye_type, eye_type,pan_position, position_updown), 'a') as f:
    #                 for line in rvecs:
    #                     #f.writelines('\n') # writing tuples
    #                     np.savetxt(f, line, delimiter=',')
    #         with open('linear_factors/{}_eye/dist_{}_pan{}_tilt{}.csv'.format(eye_type, eye_type,pan_position, position_updown), 'a') as f:
    #                 for line in dist:
    #                     #f.writelines('\n') # writing tuples
    #                     np.savetxt(f, line, delimiter=',')
    #     else:
    #         quit()


    
