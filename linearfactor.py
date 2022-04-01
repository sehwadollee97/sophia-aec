from CameraMatrix import CameraMatrix
import glob
import csv
import pandas as pd
import regex as re
import numpy as np
import yaml


camat = CameraMatrix()


class linearfactor:
    def __init__(self):
        pass

    def dataprocessing(self, filename):
        self.filename = filename
        dataset = []
        filename = glob.glob('{}'.format(filename))
        print(filename)

        mat_pan = []
        mat_tilt = []
        x_mat = np.empty((53, 0))
        y_mat = np.empty((53, 0))
        zero_coord_x = np.empty((53, 0))
        zero_coord_y = np.empty((53, 0))
        tilt=[]
        pan=[]

        # finding pan ad tilt from the file name
        for i in filename:
            regex = re.compile(r'-?\d+')  # both positive por negative integers
            regex.findall(i)
            s = [int(x) for x in regex.findall(i)]
            # print(s)
           # print

            t = s[1]
            p = s[0]

            tilt.append(t)
            pan.append(p)
        print('tilt: \n', tilt)
        print('pan: \n', pan)

        print('tilt sorted: \n', np.sort(tilt))
        print('pan sorted: \n', np.sort(pan))

        #find the least number
        pan_min=np.median(pan)
        tilt_min=np.median(tilt)

        print('pan_min', pan_min)
        print('tilt_min', tilt_min)

        #subtract the pan min and tilt min from the pan & tilt

        pan=pan-pan_min
        tilt=tilt-tilt_min
        pan=pan.astype(int)
        tilt=tilt.astype(int)

        print('tilt: \n', tilt)
        print('pan: \n', pan)


                
        for i in filename:
            #loading csv data
            print(filename)
            data = pd.read_csv('{}'.format(i))
            #print('data: \n', data)
            data = np.array(data)
            #print('data: \n', data)
            print('data: \n', data.shape)
            # extracting x
            x = data[:, 0]
            x = x.reshape(-1, 1)

            # extracting y
            y = data[:, 1]
            y = y.reshape(-1, 1)

            # appending x and y matrixes

            x_mat=np.append(x_mat, x, axis=1)
            y_mat=np.append(y_mat, y, axis=1)
        print('x_mat', x_mat)
        print('y_mat',y_mat)
        print('shape of x_mat', x_mat.shape)
        print('shape of y_mat', y_mat.shape)

        print('mat_pan: \n', x_mat[:, 7])
        print('mat_tilt: \n', y_mat[:, 7])

        for u in range(9):
            x_mat[:, u]=x_mat[:, u]-x_mat[:, 7]
            y_mat[:, u]=y_mat[:, u]-y_mat[:, 7]

        print('x_mat', x_mat[:, 7])
        print('x_mat', x_mat.shape)
        print(np.delete(x_mat, 7, axis=1))
        x_mat=np.delete(x_mat, 7, axis=1)
        _mat=np.delete(y_mat, 7, axis=1)
        print(x_mat.shape)


       
        #np.delete()
        
            
            # for j in range(9):
            #     print('tilt: \n', tilt)
            #     print('pan: \n', pan)

            #     if [pan[j], tilt[j]] != [0, 0]:
            #         mat_pan.append(pan[j])
            #         print('pan[w]',pan[j] )
            #         print('mat_pan', mat_pan)
            #         mat_tilt.append(tilt[j])
            #         print('(tilt[w]',tilt[j] )
            #         print('mat_tilt', mat_tilt)

            #         x_mat = np.append(x_mat, x, axis=1)
            #         y_mat = np.append(y_mat, y, axis=1)
            #         # print('mat_pan: \n', mat_pan)
            #         # print('mat_tilt: \n', mat_tilt)
            #     else:
            #         zero_coord_x = np.append(zero_coord_x, x, axis=1)
            #         zero_coord_y = np.append(zero_coord_y, y, axis=1)
        

            
        # compute delta
        #PT = np.vstack([mat_pan, mat_tilt])
        # print('PT, delta image \n', PT)
        # print('shape of delta_image', np.shape(PT)) # shape 2 by 9

        # print('concatenated x mat: ', x_mat)
        # print('shape of concatenated x mat: ', np.shape(x_mat))
        # print('concatenated y mat: ', y_mat)
        # print('shape of concatenated y mat: ', np.shape(y_mat))
        # print('zero_coord_x', zero_coord_x)
        # print('zero_coord_y', zero_coord_y)

        # #deleting unnecessary strings        PT = np.vstack([mat_pan, mat_tilt])

        # for i in filename:
        #     with open('{}'.format(i),'r' ) as f:
        #         data=f.read().replace('[[', '')
        #         data1=data.replace(']]', '')
        #         data2=data1.rstrip().lstrip()

        #     # print(i)

        #     with open('{}'.format(i),'w' ) as file:
        #         file.writelines(data2)
        # csvreader=csv.reader(i)

        # filename_files.append(data2)
        # print(filename_files)

        #return x_mat, y_mat, zero_coord_x, zero_coord_y, PT

    def ExtractRowOfInterst(self, x, y, index, zero_coord_x, zero_coord_y):
        x_index = x[index, :]
        y_index = y[index, :]

        zero_coord_x_array = np.repeat(zero_coord_x[index], 8)
        zero_coord_y_array = np.repeat(zero_coord_y[index], 8)

        delta_x = x_index-zero_coord_x_array
        delta_y = y_index-zero_coord_y_array

        # print('x_index \n', x_index)
        # print('zero_coord_x_array \n', zero_coord_x_array)
        # print('delta_x \n', delta_x)

        # print('y_index \n', y_index)
        # print('zero_coord_y_array \n', zero_coord_y_array)
        # print('delta_y \n', delta_y)

        delta_image = np.vstack([delta_x, delta_y])
        print('delta image \n', delta_image)
        print('shape \n', np.shape(delta_image))

        return delta_image


if __name__ == '__main__':

    lf = linearfactor()
    N = 1
    left = 'left'
    right = 'right'
    eye_type = left
    linear_factors_left = []
    empty_pan=np.array([])
    empty_tilt=np.array([])
    pan=np.array([-40,-40,-40,-40,-40,-20,-20,-20,-20,-20,0,0,0,0,0,20,20,20,20,20,40,40,40,40,40])
    tilt=np.array([-40,-20,0,20,40,-40,-20,0,20,40,-40,-20,0,20,40,-40,-20,0,20,40,-40,-20,0,20,40])
    pan=pan.reshape((5,5))
    tilt=tilt.reshape((5,5))

    for i in range(5):
        for j in range(5):
            empty_pan=np.append(empty_pan,pan[i][j])
            empty_tilt=np.append(empty_tilt,tilt[i][j])


    print(empty_pan)
    print(empty_tilt)
    print(empty_pan.reshape((5,5)))
    print(empty_tilt.reshape((5,5)))
    
    # print('pan: \n', pan)
    # print('tilt: \n', tilt)


    # pan=pan.reshape((5,5))
    # tilt=tilt.reshape((5,5))

    # print(pan)
    # print(tilt)


    if eye_type == left:

        for b in range(53):
            filename = 'linear_factors/{}_eye/left_*.csv'.format(left)
        # remove unwanted strings
            x_mat, y_mat, zero_coord_x, zero_coord_y, PT = lf.dataprocessing(
                filename)

            # print('x_mat;  \n', x_mat)
            # print('y_mat :   \n', y_mat)
            # print(np.shape(x_mat))
            # print(np.shape(y_mat))

            delta_image = lf.ExtractRowOfInterst(
                x_mat, y_mat, b, zero_coord_x, zero_coord_y)

            # compute the pinv

            X = np.linalg.pinv(delta_image)
            print('X', X)
            print('shape  of X', np.shape(X))

            print(PT)
            lin_fact = np.dot(PT, X)
            print('lin_fact: \n', lin_fact)

           
            # testing

            #int(np.dot(lin_fact, np.hstack(x_mat[1, :], y_mat[1, :])))

            # saving linear factors
            # linear_factors_left.append(lin_fact)
            # print(b)
            # print('\n')
            # print('linear_factors_left \n', linear_factors_left)

            # linear_factors_left.append(lin_fact)


            with open('linear_factors_{}.csv'.format(left), 'a') as f:
                f.write('\n')
                f.write('{}'.format(b))
                f.write('\n')
                np.savetxt(f, lin_fact, delimiter=',')
                f.write('\n')

                # f.writelines('\n')
                # f.writelines('{}'.format(lin_fact))
                # f.writelines('\n')

    # with open('linear_factors_{}.txt'.format(left), 'r') as f:
    #     for line in f:
    #         print(line)
    linear_factor = pd.read_csv('linear_factors_{}.csv'.format(left))
    print('linear_factor  \n', linear_factor)

    tester = np.array([[-0.20201365, - 0.1384257], [-0.03769018,  0.60039713]])

    # getting the linear factor of the desired coordinates

    if eye_type == right:

        for b in range(53):
            filename = 'linear_factors/{}_eye/*.csv'.format(right)
        # remove unwanted strings
            x_mat, y_mat, zero_coord_x, zero_coord_y, PT = lf.dataprocessing(
                filename)

            print('x_mat;  \n', x_mat)
            print('y_mat :   \n', y_mat)
            print(np.shape(x_mat))
            print(np.shape(y_mat))

            delta_image = lf.ExtractRowOfInterst(
                x_mat, y_mat, b, zero_coord_x, zero_coord_y)

            # compute the pinv

            X = np.linalg.pinv(delta_image)
            print('X', X)
            print('shape  of X', np.shape(X))

            print(PT)
            lin_fact = np.dot(PT, X)
            print('lin_fact: \n', lin_fact)

            with open('linear_factors_{}.txt'.format(right), 'a') as f:
                f.writelines('\n')
                f.writelines('{}'.format(b))
                f.writelines('\n')
                f.writelines('{}'.format(lin_fact))
                f.writelines('\n')

    # def obtain_PT(self, filename):
    #     filename=glob.glob('{}'.format(filename))
    #     mat_pan=[]
    #     mat_tilt=[]

    #     #print(self.filenames)
    #     for i in filename:
    #         # print(i)
    #         # print('data type', type(i))

    #         regex = re.compile(r'-?\d+')  # both positive por negative integers
    #         regex.findall(i)
    #         s=[int(x) for x in regex.findall(i)]

    #         tilt=s[1]
    #         pan=s[0]
    #         mat_pan.append(pan)
    #         mat_tilt.append(tilt)
    #         #print('mat_pan: \n', mat_pan)
    #         #print('mat_tilt: \n', mat_tilt)

    #     print('mat_pan \n', mat_pan)
    #     print('mat_tilt \n', mat_tilt)
    #     #compute delta
    #     PT=np.vstack([mat_pan, mat_tilt])
    #     print('PT, delta image \n', PT)
    #     print('shape of delta_image', np.shape(PT)) # shape 2 by 9

    #     #delta image completed
    #     #find [0;0]

    #     self.idx0=np.argwhere(PT[0, :]==0)
    #     self.idx1=np.argwhere(PT[1, :]==0)

    #     print(self.idx0)
    #     print(self.idx1)
    #     print(np.intersect1d(self.idx0, self.idx1))

    #     self.zero_idx=np.intersect1d(self.idx0, self.idx1)

    #     # locate the [0;0] from the delta_image
    #     #print(np.array([PT(0, zero_idx), PT(1, zero_idx)]))
    #     #print(PT[PT==0])
    #     #print(PT[PT[1, :]==0])
    #     print(PT[:, self.zero_idx])
    #     print('shape of zero index:', np.shape(PT[:, self.zero_idx]))

    #     print(np.delete(PT, self.zero_idx, axis=1))
    #     PT=np.delete(PT, self.zero_idx, axis=1)
    #     print(PT)

    #     return PT

    # def CreateXY(self, x_mat, y_mat):

    #     #find the x and y coordinates at the center 0,0

    #     #0,0
    #     print('original x coordinates: ', x_mat[:, 6])
    #     print('original y coordinates: ', y_mat[:, 6])
    #     orig_x=[]
    #     orig_y=[]

    #     for t in range(9):
    #         xx=x_mat[:, t]-x_mat[:, 6]
    #         orig_x.append(xx)
    #     orig_x=np.array(orig_x)
    #     orig_x=orig_x.T
    #     print('delta x orig_x: ', orig_x)
    #     print('shape of delta x orig_x: ', np.shape(orig_x))

    #     for t in range(9):
    #         yy=y_mat[:, t]-y_mat[:, 6]
    #         orig_y.append(yy)
    #     orig_y=np.array(orig_y)
    #     orig_y=orig_y.T
    #     print('delta y orig_y: ', orig_y)
    #     print('shape of delta y orig_y: ', np.shape(orig_y))

    #             #x_matrix=np.hstack(x)

    #     return self.filename, orig_x, orig_y

    # X=np.linalg.pinv()
    # print('X', X)
    # print('shape  of X', np.shape(X))

    # compute pseudo inverse
    # param=np.dot( delta_pt,X)
    # print('param', param)

    # with open('linear_factors_{}.txt'.format(left), 'a') as f:
    #     f.writelines('\n')
    #     f.writelines('{}'.format(a))
    #     f.writelines('\n')
    #     f.writelines('{}'.format(param))
    #     f.writelines('\n')

    # delta image completed
    # find [0;0]

    # self.idx0=np.argwhere(PT[0, :]==0)
    # self.idx1=np.argwhere(PT[1, :]==0)

    # print(self.idx0)
    # print(self.idx1)
    # print(np.intersect1d(self.idx0, self.idx1))

    # self.zero_idx=np.intersect1d(self.idx0, self.idx1)

    # # locate the [0;0] from the delta_image
    # #print(np.array([PT(0, zero_idx), PT(1, zero_idx)]))
    # #print(PT[PT==0])
    # #print(PT[PT[1, :]==0])
    # print(PT[:, self.zero_idx])
    # print('shape of zero index:', np.shape(PT[:, self.zero_idx]))

    # print(np.delete(PT, self.zero_idx, axis=1))
    # PT=np.delete(PT, self.zero_idx, axis=1)
    # print(PT)
