import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import pandas as pd


eye_type='left'
filename = 'linear_factors/{}_eye/{}_pan463_tilt731.png'.format(eye_type, eye_type)

data=pd.read_csv('linear_factors/{}_eye/{}_pan463_tilt731.csv'.format(eye_type, eye_type))
img=image.imread(filename)
print(data)
data=np.array(data)

print('x: \n', (data[0, 0]-data[52, 0])/8)
print('y: \n',1.636032257080078125e+02-data[4, 1])
plt.figure()
plt.imshow(img)
plt.show()

