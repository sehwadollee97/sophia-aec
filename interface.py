
# interface-imports classes of robots
# instantiates the robots
# instantiates the agent
# write the base class


try:
    #import pickle
    import time
    import os
    from os import stat
    import scipy.io as scio
    import matplotlib.pyplot as plt
    #import matplotlib.pyplot as plt
    import sys
    import numpy as np
    import math
    import cv2
    from PIL import Image
    from sophia import Sophia
    from Agent import Agent
    #from PygameRobot import PygameRobot
    import pygame
    from CamControl import CamControl
    from EyeMotor import EyeMotor
    from AECControl import AECControl
    import matplotlib.patches as patches
    #filepath = os.path.dirname(os.path.realpath(__file__))
    #print('filepath', filepath)

    filepath='/home'

    from AEC_vision.aec_vision.utils.simulation import Simulation
    from sklearn.metrics import mean_squared_error
    from AEC_vision.aec_vision.utils.environment import Environment
    from AEC_vision.aec_vision.utils.utils import get_cropped_image
    from AEC_vision.aec_vision.utils.plots import plot_observations
    from AEC_vision.aec_vision.utils.graphics_math.constant_tools import deg_to_rad, rad_to_deg
    
    #print(os.path.join(filepath,'/fyp/Downloads/SCServo_Python_200831/SCServo_Python',"AEC-vision", "/aec-vision/", "utils"))

except Exception as e:
    print('Some modules are missing {}'.format(e))

filepath = os.path.dirname(os.path.realpath(__file__))
print('filepath', filepath)

# decicde robot
robotype = 3

patch_size = 150

# instantiate agent
agent = Agent(10)

camctrl = CamControl()  # instantiate the cameral control

left = Sophia(14)
right = Sophia(15)
updown = Sophia(16)

# from camcalib class
pan_left_scale_a=0.3693410005694254
pan_left_scale_b=-0.8143205148520528
pan_right_scale_a=0.669696667088757
pan_right_scale_b=-0.33711135107519763

tilt_left_scale_a=-1.4229887456986126
tilt_left_scale_b=1.8853966426007918
tilt_right_scale_a=-1.0942264472850682
tilt_right_scale_b=1.7816286633368563

####################################################################################
#calibration

if robotype==2:



    while True:

        #image generation from the robot

        texture_dist = 500
        simulation = Simulation(action_to_angle=[-1,0,1])
        simulation.environment.add_texture(dist=texture_dist)
        img, _ = simulation.environment.render_scene()
        # plt.imshow(img)
        # plt.show()

        simulation.new_episode(texture_dist=texture_dist, texture_file='data/texture1.jpg')
        img1, _ = simulation.environment.render_scene()
        # plt.imshow(img1)
        # plt.show()

        img_dim_x, img_dim_y=agent.ImageProcessing(img1, patch_size, 2)

        
        print('image dimension x_right:  ', img_dim_x)
        print('image dimension y_right:  ', img_dim_y)

        # generate random coordinates on the image  & ensures
        agent.generate_randomTargetCoord()

        pan, tilt= agent.generate_eyeCmd(pan_left_scale_a, pan_left_scale_b, pan_right_scale_a,pan_right_scale_b,
        tilt_left_scale_a,tilt_left_scale_b,  tilt_right_scale_a, tilt_right_scale_b)

        # for now, 
        # keep the tilt=0
    
        pan_min=-2500
        pan_max=2500
        tilt_min=-2500
        tilt_max=2500

        camera_angle = 0.5
        pan_valid, tilt_valid=simulation.check_the_limits(camera_angle, tilt,pan_min, pan_max, tilt_min, tilt_max)
        agent.compute_saccade_positions(pan_valid, tilt_valid)
        # show image center
        #show the tx_ty position 
        #caqpture the image 
        agent.plot_images(img1)
        img_left_fine, img_left_coarse, img_right_fine, img_right_coarse = simulation.get_observations()

        simulation.environment.move_camera(camera_angle)
        print('after moving camera')
        #show image center
        # show how much camera angle rotation leads to how much shift in the image pixel
        # 
        #agent.plot_images(img1)


        # have to capture the image around the 
        img_left_fine, img_left_coarse, img_right_fine, img_right_coarse = simulation.get_observations()


        disparity = mean_squared_error(img_left_fine, img_right_fine)
        plot_observations(img_left_fine, img_right_fine, texture_dist=texture_dist,
                          camera_angle=simulation.environment.camera_angle, disparity=disparity)

if robotype==3:
    left_ab_t=np.array([[0.], [0.]])
    # average cd: 
    left_cd_t=np.array([[-1.2786291033056958],[1.7738629432174622]])
    
    # left pan values
    # average ab: 
    left_ab_p=np.array([[-0.6001669153098366], [-3.519050592103776]])
    # average cd: 
    left_cd_p=np.array([[0.], [0.]])
    # right tilt values
    # average ab: 
    right_ab_t=np.array([[0.], [0.]])
    # average cd: 
    right_cd_t=np.array( [[ 30.81630810102735], [ 3.651439735298863]])
    # right pan values
    # average ab: 
    right_ab_p=np.array([[-0.6004150404291795], [ 2.7404283886457885]])
    # average cd: 
    right_cd_p=np.array([[0.], [0.]])
    
    # min_left = left.calibration(0)
    # max_left = left.calibration(1)
    # left.move((min_left+max_left)/2)

    _, _, mid_l=left.get_limit()
    left.move(mid_l)
    _, _, mid_r=right.get_limit()
    right.move(mid_r)
    min_u, max_u, mid_u=updown.get_limit()
    updown.move(mid_u)

    #obain the current position of the left and right
    left_currentpos=left.get_currentPos()
    right_currentpos=right.get_currentPos()
    updown_currentpos=updown.get_currentPos()

    # aec = AECControl()
    # test without using AECControl class
    
    import time
    eye_type='Right'

    while True:
        if eye_type=='Left':
            left_cap = cv2.VideoCapture(0)
            ret, left_img = left_cap.read()
            #left_img = cv2.resize(left_img,(320,240), interpolation=cv2.INTER_NEAREST)

            # plt.figure()
            # plt.imshow(left_img)
            # plt.title('original image')
            # plt.show()
            left_cap.release()

    
            #current position of left and right eye
            currPos_left=left.get_currentPos()
            #currPos_right=right.get_currentPos()
            currPos_updown=updown.get_currentPos()

            #getting min, max and mid values for left and right eyes-pan & tilt

            min_left, max_left, mid_left=left.get_limit()
            #min_right, max_right, mid_right=right.get_limit()
            min_updown, max_updown, mid_updown=updown.get_limit()

            #!!!!!!!!!!!!!!!!! have to decide left or right
            img_dim_x, img_dim_y=agent.ImageProcessing(left_img, patch_size, robotype, eye_type)
            print('image width:  ', img_dim_x)
            print('image height:  ', img_dim_y)

            tx_list=[100, 200, 300, 400, 500]
            ty_list=int(img_dim_y/2)

            # for i in range(len(tx_list)):
            #     t_x=tx_list[i]
            #     t_y=ty_list
            #     print('t_x, t_y: ',t_x, t_y)
            #     time.sleep(2)

            # generate random coordinates on the image  & ensures
            agent.generate_randomTargetCoord()
            pan, tilt, tx, ty= agent.generate_eyeCmd(left_ab_t, left_cd_t, right_ab_t, right_cd_t, 
            left_ab_p, left_cd_p, right_ab_p, right_cd_p)

            #testing



            #setting the value of the pan_max,min and tilt max and min
            pan_min = -200
            pan_max = 200
            tilt_min = -200
            tilt_max = 200


            pan_valid, tilt_valid =left.check_the_limits(
                pan, tilt,pan_min, pan_max, tilt_min, tilt_max)


            pan_valid, tilt_valid=agent.compute_saccade_positions(pan_valid, tilt_valid)

            print('directional pan_valid:  ', pan_valid)
            print('directional tilt_valid:  ', tilt_valid)


            print('moving the pan')
            cur_pos_left=left.moveby(pan_valid) # move by 


            print('moving the tilt')
            cur_pos_updown=updown.moveby(tilt_valid)

            print('capture!')
            print('tx:  ', tx)
            print('ty:  ', ty)
            print('directional pan_valid:  ', pan_valid)
            print('directional tilt_valid:  ', tilt_valid)
            
            time.sleep(2)
            left_cap_as = cv2.VideoCapture(0)
            ret, left_img_as = left_cap_as.read()
            #left_img_as = cv2.resize(left_img_as,(320,240), interpolation=cv2.INTER_NEAREST)

            # plt.figure()
            # plt.suptitle('as')
            # plt.imshow(left_img_as)
            # plt.show()
            left_cap_as.release()
            agent.plot_images(left_img_as, currPos_left, currPos_updown)
            time.sleep(2)

        if eye_type=='Right':
            
            right_cap = cv2.VideoCapture(2)
            ret, right_img = right_cap.read()
            # image capture
            #right_img = cv2.resize(right_img,(320,240), interpolation=cv2.INTER_NEAREST)
            plt.figure()
            #plt.imshow(right_img)
            
            #plt.show()
            # aec.right.release()
            # aec.left.release()
            right_cap.release()
        

            #current position of left and right eye
            currPos_left=left.get_currentPos()
            currPos_right=right.get_currentPos()
            currPos_updown=updown.get_currentPos()

            #getting min, max and mid values for left and right eyes-pan & tilt

            min_left, max_left, mid_left=left.get_limit()
            min_right, max_right, mid_right=right.get_limit()
            min_updown, max_updown, mid_updown=updown.get_limit()

            #!!!!!!!!!!!!!!!!! have to decide left or right
            img_dim_x, img_dim_y=agent.ImageProcessing(right_img, patch_size, robotype, eye_type)
            print('image dimension x_right:  ', img_dim_x)
            print('image dimension y_right:  ', img_dim_y)


    #     left tilt values
    # average ab: 


            # generate random coordinates on the image  & ensures
            agent.generate_randomTargetCoord()
            pan, tilt, tx, ty= agent.generate_eyeCmd(left_ab_t, left_cd_t, right_ab_t, right_cd_t, 
            left_ab_p, left_cd_p, right_ab_p, right_cd_p)

            print('pan:   ', pan)


            print('tilt:   ', tilt)

            #setting the value of the pan_max,min and tilt max and min
            pan_min = -200
            pan_max = 200
            tilt_min = -200
            tilt_max = 200

            # veryfying the limits of he left eye in sophia
            # pan_valid, tilt_valid = Left_eye.check_the_limits(
            #     pan, tilt,cur_pos_left, cur_pos_updown, min_left, max_left, min_updown, max_updown)

            pan_valid, tilt_valid =right.check_the_limits(
                pan, tilt,pan_min, pan_max, tilt_min, tilt_max)
            
            # pan_valid, tilt_valid = left.check_the_limits(
            #     pan, tilt,pan_min, pan_max, tilt_min, tilt_max)


            pan_valid, tilt_valid=agent.compute_saccade_positions(pan_valid, tilt_valid)

            print('moving the pan')
            cur_pos_left=right.moveby(pan_valid) # move by 

            print('moving the tilt')
            cur_pos_updown=updown.moveby(tilt_valid)
            print('pan, tilt: ',pan_valid, tilt_valid)

            time.sleep(2)
            right_cap = cv2.VideoCapture(2)
            ret, right_img_as = right_cap.read()
            #right_img_as = cv2.resize(right_img_as,(320,240), interpolation=cv2.INTER_NEAREST)
            right_cap.release()
            

            agent.plot_images(right_img_as, currPos_right, currPos_updown)
            time.sleep(2)

            #print('correlation coefficient:  ', corr)
            right_cap.release()
            #left_cap.release()
            



