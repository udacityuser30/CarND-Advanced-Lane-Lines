"""
Created on Thu Mar 26 18:39:59 2020

@author: Sergio Marin
"""
def Camera_Calibration():
    #importing some useful packages
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    import cv2
    import os
    #%matplotlib inline
    #Now let's check our images for the camera calibration
    images = os.listdir("camera_cal/")
        
    #Let's find the corners of the chessboard
    # prepare object points
    nx = 9
    ny = 6
    objpoints = [] #3D points in real world space
    imgpoints = [] #2D points  in image plane
    
    # Convert the images to grayscale
    for image in images:
        image_read = mpimg.imread("camera_cal/"+image)
        gray = cv2.cvtColor(image_read, cv2.COLOR_RGB2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # Prepare the objpoints and imgpoints arrays
        #Let's prepare the objpoints with the structure (0,0,0), (1,0,0), (2,0,0),..., (8,5,0)
        objp = np.zeros((9*6,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)   
        
        # If found, draw corners
        if ret == True:
            # Draw and display the corners
            imgpoints.append(corners)
            objpoints.append(objp)
            cv2.drawChessboardCorners(image_read, (nx, ny), corners, ret)
    
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_read.shape[1::-1],None, None)
    print("The value of mtx is"+str(mtx))
    print("The value of dist is"+str(dist))
    return (mtx,dist)
        
   # for image in images:
        #image_read = mpimg.imread("camera_cal/"+image)
        #undist = cv2.undistort(image_read, mtx, dist, None, mtx)
        #f, (ax1, ax2) = plt.subplots(1, 2, figsize = (20,10))
        #ax1.set_title('Source image')
        #ax1.imshow(image_read)
        #ax2.set_title('Undistorted image')
        #ax2.imshow(undist)   
    
