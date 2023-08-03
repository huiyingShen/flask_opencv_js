#test-solvePnP.py

import numpy as np
import cv2

#camera calibration matrix and distortion coefficients (vector):
mtx = np.array([[1.37619399e+03, 0.00000000e+00, 9.77658017e+02],[0.00000000e+00, 1.38027871e+03, 5.10537545e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([ 0.11008736, -0.2437555 ,  0.0013639 , -0.00133615,  0.11958776])

n = 4 #number of points
#note strange feature of Python bindings to OpenCV: objpoints must have dimensions n x 3 x 1 for n points, and imgpoints must have dimensions n x 2 x 1
objpoints = np.zeros((n,3,1),float)
imgpoints = np.zeros((n,2,1),float)

#objpoints contains the four corners of a square along the z=0 plane:
objpoints[0,0,0] = 0.
objpoints[0,1,0] = 1.
objpoints[0,2,0] = 0.

objpoints[1,0,0] = 1.
objpoints[1,1,0] = 1.
objpoints[1,2,0] = 0.

objpoints[2,0,0] = 1.
objpoints[2,1,0] = 0.
objpoints[2,2,0] = 0.

objpoints[3,0,0] = 0.
objpoints[3,1,0] = 0.
objpoints[3,2,0] = 0.

#imgpoints contains the four points where the corners appear in the image:
imgpoints[0,0,0] = 0.
imgpoints[0,1,0] = 10.

imgpoints[1,0,0] = 10.
imgpoints[1,1,0] = 11.

imgpoints[2,0,0] = 10.
imgpoints[2,1,0] = 0.

imgpoints[3,0,0] = 0.
imgpoints[3,1,0] = 0.

ret, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, mtx, dist)

print('rvec:',rvec)
print('tvec:',tvec)
