from audioop import cross
from errno import EKEYEXPIRED
from os import access
from re import A
from statistics import mode
from tkinter import Image
from tkinter.font import names
from unicodedata import decimal
from cv2 import sqrt
import cv2
from scipy import rand
import numpy.matlib
import numpy.ma as ma
from sklearn.preprocessing import PolynomialFeatures
#from CameraMatrix import CameraMatrix
import glob
import csv
import pandas as pd
import regex as re
import numpy as np
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
import numpy_indexed as npi
from numpy.random import seed
from numpy.random import randint
from numpy import loadtxt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import ShuffleSplit

from platform import python_version

print(python_version())



#camat = CameraMatrix()

class linearfactor:
    def __init__(self):
        pass

    def createTestdata(self, inputdata, i, j):
        #a=np.arange(25).reshape(5, 5)
        #print(a)
        #b=0
        
        train=inputdata[i:i+3, j:j+3]
        print(train)
        b=inputdata.copy()
        b[i:i+3, j:j+3]=999

        #a[i:i+3, j:j+3]=999
        print(b)
        b=b.reshape(1,-1)
        b=b.tolist()
        b=b[0]
        print(b)
        p=[value for value in b if value !=999]
        p=np.array([p])
        p=p.reshape(1, -1)
        print(p)
        print(p.shape)
        return p

    def createTestDataforModel(self, data_x, data_y,matA,  sizeofTest):
        datasize=data_x.shape[1]
        print(datasize)

        no_iter=datasize-sizeofTest+1
        print(no_iter)
        testdata=[]

        for i in range(no_iter):
            result_x=data_x[:, i:i+sizeofTest]
            print(result_x)
            result_y=data_y[:, i:i+sizeofTest]
            print(result_y)

            inter=np.vstack([result_x, result_y])
            result_test=np.dot(matA, inter)
            testdata.append(result_test)
            #testdata.append(result_x)

        testdata=np.array([testdata])
        print(testdata.shape)
        testdata=np.reshape(testdata, (no_iter,2, sizeofTest))
        print(testdata)

        return testdata

    def createTestDataforModel2(self, data_x, data_y,  sizeofTest):
        datasize=data_x.shape[1]
        print(datasize)

        no_iter=datasize-sizeofTest+1
        print(no_iter)
        testdata=[]

        for i in range(no_iter):
            result_x=data_x[:, i:i+sizeofTest]
            print(result_x)
            result_y=data_y[:, i:i+sizeofTest]
            print(result_y)

            inter=np.vstack([result_x, result_y])
            #result_test=np.dot(matA, inter)
            testdata.append(inter)
            #testdata.append(result_x)

        testdata=np.array([testdata])
        print(testdata.shape)
        testdata=np.reshape(testdata, (no_iter,2, sizeofTest))
        print(testdata)

        return testdata


    def display_multiple_img(self, images, rows = 1, cols=1):
        figure, ax = plt.subplots(nrows=rows,ncols=cols )
        for ind,title in enumerate(images):
            ax.ravel()[ind].imshow(images[title])
            ax.ravel()[ind].set_title(title)
            ax.ravel()[ind].set_axis_off()
        plt.tight_layout()
       # plt.show()


if __name__ == '__main__':

    lf = linearfactor()
    filetype='csv'
    N = 1
    left = 'left'
    right = 'right'
    Readmode='NONE'
    testmode='testmode'
    mlmode='mlmode'
    findminmode='findminmode'

    current_mode=Readmode
    eye_type =right

        # First create some toy data:
    x = np.linspace(0, 2*np.pi, 400)
    y = np.sin(x**2)

    # # Create just a figure and only one subplot
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # ax.set_title('Simple plot')
    # plt.show()

    # # Create two subplots and unpack the output array immediately
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.plot(x, y)
    # ax1.set_title('Sharing Y axis')
    # ax2.scatter(x, y)
    # plt.show()

    # # Create four polar axes and access them through the returned array
    # fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
    # axs[0, 0].plot(x, y)
    # axs[1, 1].scatter(x, y)

    # # Share a X axis with each column of subplots
    # plt.subplots(2, 2, sharex='col')
    # plt.show()

    # # Share a Y axis with each row of subplots
    # plt.subplots(2, 2, sharey='row')

    # # Share both X and Y axes with all subplots
    # plt.subplots(2, 2, sharex='all', sharey='all')

    # # Note that this is the same as
    # plt.subplots(2, 2, sharex=True, sharey=True)

    # Create figure number 10 with a single subplot
    # and clears it if it already exists.
    fig, ax = plt.subplots(2, 5)
    pan=[-40, -20, 0, 20, 40]
    #plt.sub
    plt.figure()
    # for i in range(len(pan)):

    #     img=mpimg.imread('linear_factors/{}_eye/images/corners/{}_-40.png'.format(eye_type,pan[i]))
    #     img1=mpimg.imread('linear_factors/{}_eye/images/corners/{}_-20.png'.format(eye_type,pan[i]))
    #     img2=mpimg.imread('linear_factors/{}_eye/images/corners/{}_0.png'.format(eye_type,pan[i]))
    #     img3=mpimg.imread('linear_factors/{}_eye/images/corners/{}_20.png'.format(eye_type,pan[i]))
    #     img4=mpimg.imread('linear_factors/{}_eye/images/corners/{}_40.png'.format(eye_type,pan[i]))
    #     #plt.figure()
        
    #     plt.subplot(5, 5, i+1)
    #     plt.imshow(img)
    #     plt.subplot(5, 5, i+1+5*1)
    #     plt.imshow(img1)
    #     plt.subplot(5, 5, i+1+5*2)
    #     plt.imshow(img2)
    #     plt.subplot(5, 5, i+1+5*3)
    #     plt.imshow(img3)
    #     plt.subplot(5, 5, i+1+5*4)
    #     plt.imshow(img4)
    # plt.savefig('linear_factors/{}_eye/images/corners/CheckerboardImg55.png'.format(eye_type))
   # plt.show()

    # plt.figure()
    # for i in range(len(pan)):

    #     img=mpimg.imread('linear_factors/{}_eye/images/{}_-40.png'.format(eye_type,pan[i]))
    #     img1=mpimg.imread('linear_factors/{}_eye/images/{}_-20.png'.format(eye_type,pan[i]))
    #     img2=mpimg.imread('linear_factors/{}_eye/images/{}_0.png'.format(eye_type,pan[i]))
    #     img3=mpimg.imread('linear_factors/{}_eye/images/{}_20.png'.format(eye_type,pan[i]))
    #     img4=mpimg.imread('linear_factors/{}_eye/images/{}_40.png'.format(eye_type,pan[i]))
    #     #plt.figure()
        
    #     plt.subplot(5, 5, i+1)
    #     plt.imshow(img)
    #     plt.subplot(5, 5, i+1+5*1)
    #     plt.imshow(img1)
    #     plt.subplot(5, 5, i+1+5*2)
    #     plt.imshow(img2)
    #     plt.subplot(5, 5, i+1+5*3)
    #     plt.imshow(img3)
    #     plt.subplot(5, 5, i+1+5*4)
    #     plt.imshow(img4)
    # plt.savefig('linear_factors/{}_eye/images/corners/OptImg55.png'.format(eye_type))
    # plt.show()
    
    if eye_type == right and current_mode==Readmode:
    


        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(5.5, 3.5),
                        constrained_layout=True)
        # add an artist, in this case a nice label in the middle...
        for row in range(2):
            for col in range(2):
                axs[row, col].annotate(f'axs[{row}, {col}]', (0.5, 0.5),
                                    transform=axs[row, col].transAxes,
                                    ha='center', va='center', fontsize=18,
                                    color='darkgrey')
        fig.suptitle('plt.subplots()')
       # fig.show()


        delta_motor=pd.read_csv('linear_factors/{}_eye/{}_deltaMotor.csv'.format(eye_type,eye_type), header=None)
        P_actual=pd.read_csv('linear_factors/{}_eye/{}_panPos.csv'.format(eye_type,eye_type),header=None)
        T_actual=pd.read_csv('linear_factors/{}_eye/{}_tiltPos.csv'.format(eye_type,eye_type),header=None)
        Coord_data=pd.read_csv('linear_factors/{}_eye/{}_Coord_data.csv'.format(eye_type,eye_type),header=None)
        Coord_data27=pd.read_csv('linear_factors/{}_eye/{}_Coord_data27.csv'.format(eye_type,eye_type),header=None)
        total_images=25
        fig = plt.figure(figsize=(100, 100))

        pan=[-40, -20, 0, 20, 40]
        tilt=[-40, -20, 0, 20, 40]
        file_neg40=[]
        columns = 5
        rows = 1
        row_ind=[1, 2, 3, 4, 5]
      
        delta_motor=np.array(delta_motor)
        P_actual=np.array(P_actual)
        T_actual=np.array(T_actual)
        Coord_data=np.array(Coord_data)

        print('delta_motor: \n', delta_motor.shape)

        print('pan Position: \n', P_actual.shape)  
        print('tilt position : \n', T_actual.shape) 
        print('Coordinate data: \n', Coord_data.shape) 
        # data pre processing
        print('delta_pan command: \n', delta_motor[:,0])
        print('delta_tilt command: \n', delta_motor[:,1])

        P_command= delta_motor[:,0]
        T_command= delta_motor[:,1]
        #  P_actual.shape

        motor_shape=int(math.sqrt(P_command.shape[0]))

        P_command=np.reshape(P_command, (motor_shape,motor_shape))
        P_command=P_command.T
        
        T_command=np.reshape( T_command, (motor_shape,motor_shape))
        T_command=T_command.T
        T_command=np.flipud(T_command)

        print('reshaped motor pan: \n',P_command)
        print('reshaped motor tilt: \n', T_command)

        # P_command=np.delete(P_command,4,1)
        # P_command=np.delete(P_command, 4,0)
        # T_command=np.delete(T_command,4,1)
        # T_command=np.delete(T_command, 4,0)
        print('reshaped motor pan: \n',P_command)
        print('reshaped motor tilt: \n', T_command)

        # print('reshaped motor pan: \n', P_command)
        # print('reshaped motor tilt: \n', T_command)

        plt.figure()
        plt.scatter(P_command, T_command)
        plt.xlabel('delta pan (motor command)')
        plt.ylabel('delta tilt (motor command)')
        plt.title('motor delta pan & tilt (commands)')
        plt.savefig('linear_factors/{}_eye/motor delta pan & tilt commands.png'.format(eye_type, eye_type))
        #plt.show()



       # P_actual=P_actual.T
        P_actual=np.reshape(P_actual, (motor_shape,motor_shape)) #5,5
        P_actual=P_actual.T
        T_actual=T_actual.reshape((motor_shape,motor_shape))
        T_actual=T_actual.T
        T_actual=np.flipud(T_actual)
        # P_actual=np.delete(P_actual, 4,0)
        # P_actual=np.delete(P_actual, 4,1)
        # T_actual=np.delete(T_actual, 4,0)
        # T_actual=np.delete(T_actual, 4,1)
        print('pan_pos: \n', P_actual)
        print('T_actual: \n', T_actual)
       
       
        #P_actual=P_actual.reshape(1, -1)
        #T_actual=T_actual.reshape(1, -1)
        print('pan_pos: \n', P_actual)  #4,4
        print('T_actual: \n', T_actual)
        #print('T_actual: \n', T_actual)
        #print('reshaped  tilt position : \n', T_actual)
        plt.figure()
        plt.scatter(P_actual, T_actual)
        plt.xlabel('pan position (motor unit)')
        plt.ylabel('tilt position (motor unit)')
        plt.title('motor pan & tilt positions')
        plt.savefig('linear_factors/{}_eye/motor pan & tilt positions.png'.format(eye_type, eye_type))
        #plt.show()


        #pre process th4 coordinate data

        no_dataset=int(Coord_data.shape[0]/54)  
        
        #coord_shape=int(math.sqrt(no_dataset))
        Coord_data=np.reshape(Coord_data, (motor_shape,motor_shape, 54, 2))
        
        print('reshaped dataset: \n', Coord_data)
        print('reshaped dataset: \n', Coord_data.shape)

        #extract the 27th coordinate from the dataset coord

        coord_interest=Coord_data[:,:,27, :]
       
        coord_x=coord_interest[:,:,0]
        coord_y=coord_interest[:,:,1]
        print('coord x: \n', coord_x)
        print('coord y: \n', coord_y)
        coord_x=coord_x.T
        coord_x=np.fliplr(coord_x)
        coord_y=coord_y.T
        coord_y=np.flipud(coord_y)
        print('coord x: \n', coord_x)
        print('coord y: \n', coord_y)

        # the pan with the largest value and the tilt with the lest value dont seem linear
        # delete one column and one row--

        # coord_x=np.delete(coord_x, 4,0)
        # coord_x=np.delete(coord_x, 4,1)
        # coord_y=np.delete(coord_y, 4,0)
        # coord_y=np.delete(coord_y, 4,1)

        print('coord x: \n', coord_x)
        print('coord y: \n', coord_y)

        #for i in range(no_dataset):

        plt.figure()
        plt.scatter(coord_x, coord_y)
        plt.xlabel('pan positions(pixel)')
        plt.ylabel('tilt (pixel)')
        plt.title('pixel delta x & y')
        plt.savefig('linear_factors/{}_eye/pixel pan & tilt.png'.format(eye_type, eye_type))
        #plt.show()

        with open('linear_factors/{}_eye/matA.csv'.format(eye_type), 'ab') as f:
            f.truncate(0)

        with open('linear_factors/{}_eye/delta_motor.csv'.format(eye_type), 'ab') as f:
            f.truncate(0)

        with open('linear_factors/{}_eye/delta_pixel.csv'.format(eye_type), 'ab') as f:
            f.truncate(0)

        with open('linear_factors/{}_eye/delta_motor_test.csv'.format(eye_type), 'ab') as f:
            f.truncate(0)

        with open('linear_factors/{}_eye/delta_pixel_test.csv'.format(eye_type, eye_type), 'ab') as f:
            f.truncate(0)
        with open('linear_factors/{}_eye/mse.csv'.format(eye_type), 'ab') as f:
            f.truncate(0)

        # get the ref pan, ref tilt, ref x and ref y matrices
    
        ref_pan=P_actual[2,2]
        ref_pan=numpy.matlib.repmat(ref_pan,5,5)
        #print(ref_pan)

        ref_tilt=T_actual[2,2]
        ref_tilt=numpy.matlib.repmat(ref_tilt, 5,5)
        #print(ref_tilt)

        ref_x=coord_x[2,2]
        ref_x=numpy.matlib.repmat(ref_x,5,5)
        #print(ref_x)
        
        ref_y=coord_y[2,2]
        ref_y=numpy.matlib.repmat(ref_y,5,5)
        #print(ref_y)

        #calculate the delta pan, delta tilt, delta x and delta y
        P_actual=-(ref_pan-P_actual)
        T_actual=-(ref_tilt-T_actual)
        X=-(ref_x-coord_x)
        Y=-(ref_y-coord_y)

        print('delta_pan: \n',P_actual)
        print('delta_tilt: \n',T_actual)
        print('delta_x: \n',X)
        print('delta_y: \n',Y)

        plt.figure()
        plt.scatter(P_actual, T_actual)
        plt.xlabel('calculated delta pan (motor unit)')
        plt.ylabel('calculated delta tilt(motor unit)')
        plt.title('calculated  motor pan & tilt positions')
        plt.savefig('linear_factors/{}_eye/calculated motor pan & tilt positions.png'.format(eye_type, eye_type))
        #plt.show()

        plt.figure()
        plt.scatter(X, Y)
        plt.xlabel('calculated delta x  (pixel unit)')
        plt.ylabel('calculated delta y (pixel unit)')
        plt.title('calculated delta pixel')
        plt.savefig('linear_factors/{}_eye/calculated delta pixel.png'.format(eye_type, eye_type))
        #plt.show()

        mse_test=[]
        mse_train=[]

        index=[0, 1, 2]
        for i in range(len(index)):
            for j in range(len(index)):
                print('index[i][j]', index[i], index[j])
                sliced_ind_x=slice(index[i],index[i]+3)
                sliced_ind_y=slice(index[j],index[j]+3)
                
                print(sliced_ind_x)
                print(sliced_ind_y)

                delta_train_x=X[sliced_ind_x, sliced_ind_y]
                delta_train_y=Y[sliced_ind_x,sliced_ind_y]
                print(X)
                print(delta_train_x)
                print(Y)
                print(delta_train_y)
               
                delta_train_x=np.reshape(delta_train_x,(1, -1))
                delta_train_y=np.reshape(delta_train_y,(1, -1))
                
                print(delta_train_x)
                print(delta_train_y)

                delta_pixel_train=np.vstack([delta_train_x, delta_train_y])
                print(delta_pixel_train)

                #### do the same for the delta motor
                ##################################################

                delta_train_pan=P_actual[sliced_ind_x, sliced_ind_y]
                delta_train_tilt=T_actual[sliced_ind_x,sliced_ind_y]
                print(P_actual)
                print(delta_train_pan)
                print(T_actual)
                print(delta_train_tilt)

                delta_train_pan=np.reshape(delta_train_pan,(1, -1))
                delta_train_tilt=np.reshape(delta_train_tilt,(1, -1))
                
                print(delta_train_pan)
                print(delta_train_tilt)

                delta_motor_train=np.vstack([delta_train_pan, delta_train_tilt])
                print(delta_motor_train)

                #COMPUTE MATRIX A
                matA=np.dot(delta_motor_train, np.linalg.pinv(delta_pixel_train))
                print('matA: \n',matA)


                ###########################################################
                delta_train_pan_command=P_command[sliced_ind_x, sliced_ind_y]
                delta_train_tilt_command=T_command[sliced_ind_x,sliced_ind_y]
                print(P_command)
                print(delta_train_pan_command)
                print(T_command)
                print(delta_train_tilt_command)

                delta_train_pan_command=np.reshape(delta_train_pan_command,(1, -1))
                delta_train_tilt_command=np.reshape(delta_train_tilt_command,(1, -1))
                
                print(delta_train_pan_command)
                print(delta_train_tilt_command)

                delta_motor_train_command=np.vstack([delta_train_pan_command, delta_train_tilt_command])
                print(delta_motor_train_command)

                #COMPUTE MATRIX A
                # matA_command=np.dot(delta_motor_train_command, np.linalg.pinv(delta_pixel_train))
                # print('matA: \n',matA_command)


                delta_motor_command=np.dot(matA, delta_pixel_train)
                mse_training=metrics.mean_squared_error(delta_motor_command, delta_motor_train)
                mse_train.append(mse_training)
                with open('linear_factors/{}_eye/matA.csv'.format(eye_type), 'ab') as f:
                    np.savetxt(f, matA, delimiter=',')


                # with open('linear_factors/{}_eye/matA_command.csv'.format(eye_type), 'ab') as f:
                #     np.savetxt(f, matA_command, delimiter=',')

                with open('linear_factors/{}_eye/delta_motor.csv'.format(eye_type), 'ab') as f:
                    np.savetxt(f, delta_motor_train, delimiter=',')

                with open('linear_factors/{}_eye/delta_pixel.csv'.format(eye_type), 'ab') as f:
                    np.savetxt(f, delta_pixel_train, delimiter=',')

                
##################################################################
###############################CREATE TESTING DATA

                sizeOfTestdata=delta_motor_train.shape[1]
                

                #create delta pixel test
                delta_test_x=lf.createTestdata(X, i, j)
                delta_test_y=lf.createTestdata(Y, i, j)
                plt.figure()
                plt.scatter(delta_test_x, delta_test_y)
                plt.title('testXY_{}{}'.format(i, j))
                plt.savefig('linear_factors/{}_eye/testXY_{}{}.png'.format(eye_type, i, j))
                #plt.show()

                print('delta_test_x: \n',delta_test_x)
                print('delta_test_y: \n',delta_test_y)

                delta_test_pan=lf.createTestdata(P_actual, i, j)
                delta_test_tilt=lf.createTestdata(T_actual, i, j)
                plt.figure()
                plt.scatter(delta_test_pan, delta_test_tilt)
                plt.title('testPanTilt_{}{}'.format(i, j))
                plt.savefig('linear_factors/{}_eye/testPanTilt_{}{}.png'.format(eye_type, i, j))
                #plt.show()
                print('delta_test_pan: \n',delta_test_pan)
                print('delta_test_ilt: \n',delta_test_tilt)


                delta_test_pixel=np.vstack([delta_test_x, delta_test_y])
                delta_test_motor=np.vstack([delta_test_pan, delta_test_tilt])

                y_pred=np.dot(matA, delta_test_pixel)

                mse=metrics.mean_squared_error(y_pred, delta_test_motor)
                mse=np.array([mse])
                with open('linear_factors/{}_eye/mse.csv'.format(eye_type), 'ab') as f:
                    np.savetxt(f, mse, delimiter=',')
                mse_test.append(mse)


                plt.figure()
                plt.scatter(y_pred[0, :], y_pred[1, :], color='Red')
                plt.scatter(delta_test_motor[0, :], delta_test_motor[1, :], cmap='blue')
                plt.title('mse: {}'.format(mse))
                plt.savefig('linear_factors/{}_eye/aftertraining_{}{}.png'.format(eye_type, i, j))
                #plt.show()
        
        print(mse_test)
        mse_test=np.array([mse_test])
        mse_test_mean=np.mean(mse_test)
        print('mse_test_mean:', mse_test_mean)
        mse_train=np.array([mse_train])
        mse_train_mean=np.mean(mse_train)
        print('mse_train_mean:', mse_train_mean)


        # #         X_train=lf.createTestDataforModel(delta_test_x,delta_test_y, matA,sizeOfTestdata)
        # #         #test_y=lf.createTestDataforModel(delta_test_y ,sizeOfTestdata)
        # #         Y_train=lf.createTestDataforModel2(delta_test_pan,delta_test_tilt, sizeOfTestdata)
        # #         X_test=lf.createTestDataforModel2(delta_test_pan, delta_test_tilt, sizeOfTestdata)
        # #         #test_tilt=lf.createTestDataforModel(delta_test_tilt ,sizeOfTestdata)

        # #         nsamples1, nx1, ny1=X_train.shape
        # #         X_train=X_train.reshape((nsamples1, nx1* ny1))
        # #         Y_train=Y_train.reshape((nsamples1, nx1* ny1))
        # #         X_test=X_test.reshape((nsamples1, nx1* ny1))
        # #         #y_pred=y_pred.reshape((nsamples1, nx1* ny1))

        # #         #X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=100)      

        # # #     #implementing model
        # # #     #1 linear model

        # #         limdl=Ridge(alpha=1.0)
        # #         limdl.fit(X_train, Y_train)

        # #         print('model equation: intercept \n', limdl.intercept_)
        # #         print('model equation: coefficients \n', zip(X_train, limdl.coef_))

        # #         Y_pred=limdl.predict(X_test)

        # #         mse=metrics.mean_squared_error(Y_pred, X_test)
        # #         mse=np.array([mse])
        # #         print('mse: \n', mse)

        # #         with open('linear_factors/{}_eye/mse.csv'.format(eye_type), 'ab') as f:
        # #             np.savetxt(f, mse, delimiter=',')

        # #         #Y_pred=np.reshape(Y_pred, (Y))

        # #         plt.figure()
        # #         plt.scatter(Y_pred[:,  0:4], Y_pred[:,  4:8], color='red')
        # #         plt.scatter(X_test[:,  0:4], X_test[:, 4:8], color='black')
        # #         plt.title("MSE: {}".format(mse))
        # #         plt.savefig('linear_factors/{}_eye/aftertraining_{}{}.png'.format(eye_type, i, j))
        # #         #plt.imshow()
        # #         #plt.show()

                


               