from cv2 import cv2 as cv 
import numpy as np
 

left_camera_matrix = np.array([[7.564981010657308e+02, -0.927468806458065, 3.303715050420852e+02],
                               [0, 7.567388637834070e+02, 1.800044991612013e+02],
                               [0, 0, 1]])
left_distortion = np.array([[0.101586615480875, 0.019875630364583, 0, 0, 0]])
 
right_camera_matrix = np.array([[7.547392133037334e+02, 1.136690100098527, 3.324149037982965e+02],
                                [0, 7.547732334204624e+02, 1.829714620687263e+02],
                                [0, 0, 1]])
right_distortion = np.array([[0.011776565463132, 0.884280553262476, 0, 0, 0]])
 
R = np.array([[0.999796032539970, 0.012805259045708, -0.015617895444287],    
              [-0.012902777372537, 0.999897785392298, -0.006159310322828],
              [0.015537427503004, 0.006359568251891, 0.999859062187386]])

T = np.array([99.413101525006240, -1.360615877694908, 3.158269031796609])     
 
size = (640, 360) 
 

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(left_camera_matrix, left_distortion, right_camera_matrix, right_distortion, size, R,T)

left_map1, left_map2 = cv.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv.CV_16SC2)
right_map1, right_map2 = cv.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv.CV_16SC2)