

try:

    import pickle
    import time
    import os
    import math
    import numpy as np
    import cv2
    import random

    import scipy.io as scio
    import matplotlib.pyplot as plt

    # environment
    from numpy.lib.shape_base import expand_dims
    from skimage.util import view_as_windows

    from CamControl import CamControl


    from typing import Counter
    import matplotlib.pyplot as plt
    import os
    import math
    from scservo_sdk import *
    import yaml
    if os.name == 'nt':                                                    #def getch() for interactive control 
        import msvcrt
        def getch():
            return msvcrt.getch().decode()
            
    else:
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        def getch():
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

    from scservo_sdk import * 
except Exception as e:
    print('Some modules are missing {}'.format(e))


class Sophia:

    def __init__(self,ID,precision = 5,load_lim = 250):							#Doing initialization for the robot motor(currently only works on SCS motor)
        SCS_ID                      = ID		# Motor ID (14 = Left, 15 = Right, 16 = UpDown)
		self.precision 				= precision		#use in calibration to determine size of each step 
		self.side 					= 0				#use in calibration
		self.load_lim 				= load_lim
		self.ID                     = ID	                # SCServo ID : 1
		self.BAUDRATE                    = 1000000           # SCServo default baudrate : 1000000
		self.DEVICENAME = DEVICENAME     = '/dev/ttyUSB3'
		self.SCS_MOVING_STATUS_THRESHOLD = 1          # SCServo moving status threshold
		self.SCS_MOVING_SPEED            = 1           # SCServo moving speed
		self.SCS_MOVING_ACC              = 1           # SCServo moving acc
		self.ADDR_SCS_TORQUE_ENABLE     = 40           #Address needed for input/output       
		self.ADDR_SCS_GOAL_ACC          = 41
		self.ADDR_SCS_GOAL_POSITION     = 42
		self.ADDR_SCS_GOAL_SPEED        = 46
		self.ADDR_SCS_PRESENT_POSITION  = 56
		self.protocol_end   = protocol_end   = 1		#Protocol_end for SCS motor
		self.portHandler = PortHandler(DEVICENAME)			#set PortHandler
		self.packetHandler = PacketHandler(protocol_end)	#set PacketHandler
		self.portHandler.openPort()							#open port	
		self.portHandler.setBaudRate(self.BAUDRATE)			#set BaudRate
		#Write accelaration
		scs_comm_result, scs_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.ID, self.ADDR_SCS_GOAL_ACC, self.SCS_MOVING_ACC)
		if scs_comm_result != COMM_SUCCESS:
			print("%s" % self.packetHandler.getTxRxResult(scs_comm_result))
		elif scs_error != 0:
			print("%s" % self.packetHandler.getRxPacketError(scs_error))

		# Write speed
		scs_comm_result, scs_error = self.packetHandler.write2ByteTxRx(self.portHandler,self.ID, self.ADDR_SCS_GOAL_SPEED, self.SCS_MOVING_SPEED)

		if scs_comm_result != COMM_SUCCESS:
			print("%s" % self.packetHandler.getTxRxResult(scs_comm_result))
		elif scs_error != 0:
			print("%s" %self.packetHandler.getRxPacketError(scs_error))
####################################

    #SEAHWA
    def read_file(self, filepath):
            filepath = os.path.dirname(os.path.realpath(__file__))
        
            
            imagedata = scio.loadmat(filepath+"/LRimage1.mat")

            return imagedata

    def check_the_limits(self, pan, tilt, pan_min, pan_max, tilt_min, tilt_max):
        #print('type of pan_min before floating:  ', type(pan_min))
        #print('type of pan_max before floating:  ', type(pan_max))
        pan_min = float(pan_min)
        pan_max = float(pan_max)
        #print('type of pan_min after floating:  ', type(pan_min))
        #print('type of pan_max after floating:  ', type(pan_max))

        if pan < pan_min or pan > pan_max:
            if pan < pan_min:
                pan_valid = pan_min
            elif pan > pan_max:
                pan_valid = pan_max

        else:
            pan_valid = pan
            print('pan is within the range')

        if tilt < tilt_min or tilt > tilt_max:
            if tilt < tilt_min:
                tilt_valid = tilt_min
            elif tilt > tilt_max:
                tilt_valid = tilt_max
        else:
            tilt_valid = tilt

        return pan_valid, tilt_valid

    def Move_the_camera(self, rx, ry, rz, tx, ty, tz):
        # insantiate the camera control
        self.camctrl = CamControl()
        camera_matrix = self.camctrl.compute_cameramat(
            rx, ry, rz, tx, ty, tz)

    ###################################

	
	def get_feedback(self): #get feedback of motor 
		# Reading 15 byte starting from address 56
		data_read, results, error = self.packetHandler.readTxRx(self.portHandler, self.ID,self.ADDR_SCS_PRESENT_POSITION , 15)
		if len(data_read) ==  15: # Separate the data 
			state = {
				'time': time.time(), # Time of feedback capture
				'position': SCS_MAKEWORD(data_read[0], data_read[1]),
				'speed':  SCS_TOHOST(SCS_MAKEWORD(data_read[2], data_read[3]),15),
				'load': SCS_MAKEWORD(data_read[4], data_read[5])/1000.0,
				'voltage': data_read[6]/10.0,
				'temperature': data_read[7],
				'status': data_read[9],
				'moving': data_read[10],
				'current': SCS_MAKEWORD(data_read[13], data_read[14]),
				}
		return state,results,error

	def move(self,position,calibration = False):	#command a position and move the motor to that position
	# The moving function will command a goal further away from the position commanded to apply enough load
	# It will stop when it reach the position or have high load
		state,scs_comm_result,scs_error = self.get_feedback()
		scs_present_position  = state['position']   #get the motor status
		#check if the position is over limit
		if calibration == False:
			min,max,mid = self.get_limit()
			if position < min:
				position = min
				print("You are moving out of limit setting position to", min)
			elif position > max:
				position = max
				print("You are moving out of limit setting position to", max)
		#Adjust the goal so that it can move to the command location exactly
		if scs_present_position > position:
			goal = position - 20
		else: 
			goal = position + 20
		# Write goal location to the motor
		scs_comm_result, scs_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.ID, self.ADDR_SCS_GOAL_POSITION, int(goal))
		if calibration == True:
			t = time.time()
		while 1: #Keep the command moving
			state,scs_comm_result,scs_error = self.get_feedback()
			scs_present_position  = state['position']   #get the motor status
			load = state['load']*1000
			
			# Load shifting
			if load > 1000 and calibration == False:
				load -= 1000
			if (load > 1000 or self.side == 1 )and calibration == True:
				load -= 1000
			print("goal: ", position,"position:",scs_present_position,"load:",load) # Print out state
			# Break if the load is too high or reaching the location
			if calibration == True:
				if load >= self.load_lim:
					break
				if time.time() - t >= 5:
					break
			if(abs(position - scs_present_position)) < self.SCS_MOVING_STATUS_THRESHOLD:
				break 

	#Try to write the calibration using move function
	#The same idea of moving the motor bit by bit but the command location is different due to the move function different
	def calibration(self,side): # code for calibration (side == 0 is minimum side == 1 is maximum)
		ID = self.ID		
		self.side = side
		self.precision = self.precision
		state ,scs_comm_result,scs_error = self.get_feedback()	# get feedback for determining the goal position	
		scs_present_position  = state['position']
		goal = scs_present_position
		while True: 
			state ,scs_comm_result,scs_error = self.get_feedback()		
			overload = self.move(goal,calibration =True)
			scs_present_position  = state['position']            		#get state
			load = state['load']*1000
			if load > 1000 or side == 1:
				load -= 1000										
			print(' goal: ',goal,' position: ', scs_present_position,  'load: ',load)
			if load > self.load_lim:
				break
			elif side == 0:
				goal = goal - self.precision
			elif side == 1:
				goal = goal + self.precision

		state ,scs_comm_result,scs_error = self.get_feedback()
		limit = state["position"]
		
		return limit
		
	'''
	#original calibration code
	def calibration(self,side):
		ID = self.ID
		load_lim = self.load_lim
		precision = self.precision
		state ,scs_comm_result,scs_error = self.get_feedback()
		position  = state['position']
		goal = position
		prev_position = position
		count = 0
			#load = 0
		while True: 
			state ,scs_comm_result,scs_error = self.get_feedback()
			position  = state['position']            

			if scs_comm_result != COMM_SUCCESS:
				continue     ### if there is a communication error get it again

				
			if abs(position - goal) <= 5:
			#if position == goal:
				state ,scs_comm_result,scs_error = self.get_feedback()
				load = state['load']
				if side == 1 or load >= 1:
				#if load >1:
					load = load-1
				if side == 0: 	     ###set new goal				
					goal -= precision 
					if goal < 0:
						goal = 0
				elif side == 1:
					goal += precision
					if goal > 4000:
						goal = 4000
			
			scs_comm_result, scs_error = self.packetHandler.write2ByteTxRx(self.portHandler, ID, self.ADDR_SCS_GOAL_POSITION,goal) 
			if prev_position == position:
				count += 1
				state ,scs_comm_result,scs_error = self.get_feedback()
				load = state['load']
				if load > 1 or side == 1:
					load = load-1
			else: 
				count = 0
				if side == 0: 
					goal = position - precision
				else:
					goal = position + precision

			if count > 5:
				if side == 0:
					goal -=precision
					if goal < 0:
						goal = 0
					if side == 1:
						goal += precision
					if goal > 4000:
						goal = 4000       
			prev_position = position
			print(' goal: ',goal,' position: ', position,  'load: ',load)
			if load >= load_lim:
				break 
				#if side == 0:
				#    scs_comm_result, scs_error = packetHandler.write2ByteTxRx(portHandler, ID, ADDR_SCS_GOAL_POSITION,0)		     ### add this line to move the motor to goal
				#if side == 1:
				#    scs_comm_result, scs_error = packetHandler.write2ByteTxRx(portHandler, ID, ADDR_SCS_GOAL_POSITION,4000)

			#if side == 0:		    ###set the limit before overload
			#limit = position #+ precision
			#elif side == 1:
			#        limit = position #- precision
			state ,scs_comm_result,scs_error = self.get_feedback()
			limit = state["position"]

			return limit
		'''	

	def write(self): #writing to yaml file with calibration
		print("calibrate min")
		min = self.calibration(0)
		print("calibrate max")
		max = self.calibration(1)
		# Write to .yaml file

		fname = "feetechsmall.yaml"
		#fname = "/home/fyp/Downloads/SCServo_Python_200831/SCServo_Python/feetechsmall.yaml"
		stream = open(fname, 'r')
		data = yaml.load(stream, Loader=yaml.FullLoader)
		if self.ID == 15:
			motor_n = "EyeTurnRight"
		elif self.ID == 14:
			motor_n = "EyeTurnLeft"
		else: 
			motor_n = "EyesUpDown"
			init = int((min + max) /2)
			data[motor_n]['min'] = min
			data[motor_n]['max'] = max
			data[motor_n]['init'] = init

			with open(fname, 'w') as yaml_file:
				yaml_file.write( yaml.dump(data, default_flow_style=False)) 

	def get_limit(self,calibrate = False): #getting the limit from yaml file with the option to calibrate 
		if calibrate == True:
			self.write()
		if self.ID == 15:
			motor_n = "EyeTurnRight"
		elif self.ID == 14:
			motor_n = "EyeTurnLeft"
		else: 
			motor_n = "EyesUpDown"
		fname = "feetechsmall.yaml"
		stream = open(fname, 'r')
		data = yaml.load(stream, Loader=yaml.FullLoader)
		min = data[motor_n]['min'] 
		max = data[motor_n]['max'] 
		mid = data[motor_n]['init'] 
		return min, max, mid

if __name__ == '__main__':
	left = EyeMotor(14)
	right = EyeMotor(15)
	updown = EyeMotor(16)
	#while True:
	#	states,result,error = left.get_feedback()
	#	print('position: ',states['position'],'load: ',states['load'])
	#min = left.calibration(0)
	#max = left.calibration(1)
	min = right.calibration(0)
	max = right.calibration(1)
	
	#print(min,max)
	#left.move(500)
	right.move((437+690)/2)
	#print(min,max)








