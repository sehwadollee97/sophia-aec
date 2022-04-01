import numpy as np
# init_pixel=np.array([178.344848632813,135.575668334961 ])
# init_motor=np.array([[464,708]])

# pixelval=np.array([ 176.61296081543, 149.158432006836])
# motorval=np.array( [464, 717])
# delta_motor=motorval-init_motor
# delta_pixel=pixelval-init_pixel

# delta_motor=np.reshape(delta_motor, (-1, 1))
# delta_pixel=np.reshape(delta_pixel, (-1, 1))

# print('delta_pixel: \n', delta_pixel)
# print('delta_motor: \n', delta_motor)

# A=np.linalg.pinv(delta_pixel)
# print('A', A)
# coeff_mat=np.dot(delta_motor, A)
# print('coeff_mat: \n', coeff_mat)


# #calculate the A inverse
# A_inv=np.linalg.pinv(coeff_mat)
# print('predicted pixel value: \n',np.dot(A_inv, delta_motor))

# init_pixel=np.array([176.61296081543,149.158432006836])
# init_motor=np.array([[464,717]])

# pixelval=np.array([ 117.033630371094, 150.09440612793])
# motorval=np.array( [509, 717])
# delta_motor=motorval-init_motor
# delta_pixel=pixelval-init_pixel

# delta_motor=np.reshape(delta_motor, (-1, 1))
# delta_pixel=np.reshape(delta_pixel, (-1, 1))

# print('delta_pixel: \n', delta_pixel)
# print('delta_motor: \n', delta_motor)

# A=np.linalg.pinv(delta_pixel)
# print('A', A)
# coeff_mat=np.dot(delta_motor, A)
# print('coeff_mat: \n', coeff_mat)


# # #calculate the A inverse
# # A_inv=np.linalg.pinv(coeff_mat)
# # print('predicted pixel value: \n',A_inv)

# A=np.array([[-0.755, 0.011], [-0.08, 0.652]])
# #calc_pixel=np.dot(A, delta_motor.T)
# #print('calc_pixel', calc_pixel)
# A_inv=np.linalg.pinv(A)
# print(A_inv)

init_pixel=np.array([1.788417053222656250e+02,9.559605407714843750e+01])
init_motor=np.array([[554,708]])

pixelval=np.array([ 1.766006011962890625e+02,1.102608032226562500e+02])
motorval=np.array( [554, 717])

print(init_pixel)
print(pixelval)
delta_motor=motorval-init_motor
delta_pixel=pixelval-init_pixel

delta_motor=np.reshape(delta_motor, (-1, 1))
delta_pixel=np.reshape(delta_pixel, (-1, 1))

print('delta_pixel: \n', delta_pixel)
print('delta_motor: \n', delta_motor)

A=np.linalg.pinv(delta_pixel)
print('A', A)
coeff_mat=np.dot(delta_motor, A)
print('coeff_mat: \n', coeff_mat)




