# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 14:16:09 2020

@author: Sergio Marin
"""


import matplotlib.pyplot as plt
import numpy as np
import cv2
from camera_calibration import Camera_Calibration
mtx, dist = Camera_Calibration()

#images = os.listdir("test_images/")
#hue_thresh_yellow_lower = np.array([10, 80, 50])
hue_thresh_yellow_lower = np.array([10, 80, 50])
hue_thresh_yellow_upper = np.array([60,255,255])
#hue_thresh_yellow_upper = np.array([60,255,255])
hue_thresh_white_lower = np.array([0, 0, 200])
hue_thresh_white_upper = np.array([150,25,255])
sx_thresh = (40, 255)
left_fit = 0
right_fit = 0
init = 0

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        #radius of curvature of the line in some units
        self.radious_of_curvature = [] 
        self.detected_lines = False
        self.distance_between_x = 0
        self.distance_center = []
        
    def reset(self):
        self.radious_of_curvature = []
        self.detected_lines = False
        self.distance_center = []
        
    def average_curve(self, curvature):
        self.radious_of_curvature.append(curvature)
        if len(self.radious_of_curvature)>=25:
            average_curvature = np.mean(self.radious_of_curvature)
            del self.radious_of_curvature[0]
        else:
            average_curvature = 0        
        return (average_curvature)
    
    def average_center(self, center):
        self.distance_center.append(center)
        if len(self.distance_center)>=25:
            distance_center_avg = np.mean(self.distance_center)
            del self.distance_center[0]
        else:
            distance_center_avg = 0        
        return (distance_center_avg)
        
left_line = Line()
right_line = Line()

def Main(image): 
    global left_fit
    global right_fit
    
    
    #Part 2: Applying thresholds 
    img = np.copy(image)
    # Convert to HLS color space and separate the channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask_yellow = cv2.inRange(hsv, hue_thresh_yellow_lower, hue_thresh_yellow_upper)
    mask_white = cv2.inRange(hsv, hue_thresh_white_lower, hue_thresh_white_upper)
    res_yellow = cv2.bitwise_and(img,img, mask = mask_yellow)
    res_white = cv2.bitwise_and(img,img, mask= mask_white)
    res = cv2.bitwise_or(res_yellow,res_white)
    res_gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    res_binary = np.zeros_like(res_gray)
    res_binary[(res_gray > 150)] = 1 #Convert to binary
    
    l_channel = hls[:,:,1]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(res_binary == 1) | (sxbinary == 1)] = 1
    
    #Part 3: Undistort and warp
    #Now let's undistort and warp our test images
    distort = cv2.undistort(combined_binary, mtx, dist, None, mtx)
    image_size = (distort.shape[1], distort.shape[0])

    # Draw and display the corners
    src = np.float32([[200,image_size[1]],[image_size[0]-200,image_size[1]],[584,455],[image_size[0]-584,455]])            
    dst = np.float32([[300,image_size[1]],[image_size[0]-300,image_size[1]],[300,0],[image_size[0]-300,0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(distort, M, image_size, flags=cv2.INTER_LINEAR)
    
    #Part 4: Obtain histogram to detect base point of the lines
    bottom_half = warped[warped.shape[0]//2:,:]
    histogram = np.sum(bottom_half, axis = 0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped, warped, warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 15
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    #Part 5.1: Finding the lanes. Reset and lost lines
    if left_line.detected_lines == False or right_line.detected_lines == False:
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
    
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window+1)*window_height
            win_y_high = warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
        
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
        
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
        
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
            left_line.detected_lines = True
            right_line.detected_lines = True
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        #Fit Polynomial
        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty
    
    #Part 5.2: Finding the lanes after they where found
    else: 
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        left_line.distance_between_x = left_fitx-right_fitx
        for number in left_line.distance_between_x:
            if number == 0:
                left_line.detected_lines = False
                right_line.detected_lines = False
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
        left_line.detected_lines = False
        right_line.detected_lines = False

    #Part 6: Calculate Curvature and distance to the center
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    #Calculate values at the bottom
    left_bottom = left_fit_cr[0]*(720*ym_per_pix)**2 + left_fit_cr[1]*720*ym_per_pix + left_fit_cr[2]
    right_bottom = right_fit_cr[0]*(720*ym_per_pix)**2 + right_fit_cr[1]*720*ym_per_pix + right_fit_cr[2]
    #left_line.radious_of_curvature = left_curverad  
    #right_line.radious_of_curvature = right_curverad
    left_curverad_average = left_line.average_curve(left_curverad)
    right_curverad_average = right_line.average_curve(right_curverad)
    
    #Calculate distance to center

    center = (640*xm_per_pix - ((left_bottom + right_bottom)/2))
    center_avg = left_line.average_center(center)
    

    #8: Sanity checks, unwarping to original image and visualization
    plt.figure()
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((warped, warped, warped))*255
    result_only_Lines = np.zeros_like(out_img)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx + 3, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx +500, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - 3, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx-500, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    left_line_yellow1 = np.array([np.transpose(np.vstack([left_fitx - 10, ploty]))])
    left_line_yellow2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + 10,ploty])))])
    left_line_yellow = np.hstack((left_line_yellow1, left_line_yellow2))
    right_line_yellow1 = np.array([np.transpose(np.vstack([right_fitx + 10, ploty]))])
    right_line_yellow2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx - 10 ,ploty])))])
    right_line_yelllow= np.hstack((right_line_yellow1, right_line_yellow2))   
    # Draw the lane onto the warped blank image
    cv2.fillPoly(result_only_Lines, np.int_([left_line_pts]), (0,255,255))
    cv2.fillPoly(result_only_Lines, np.int_([right_line_pts]), (0,255,255))
    cv2.fillPoly(result_only_Lines, np.int_([left_line_yellow]), (255,0,0))
    cv2.fillPoly(result_only_Lines, np.int_([right_line_yelllow]), (255,0,0))

    image_size = (result_only_Lines.shape[1], result_only_Lines.shape[0])
    #Now let's undistort and warp our test images
    dst = np.float32([[200,image_size[1]],[image_size[0]-200,image_size[1]],[584,455],[image_size[0]-584,455]])            
    src = np.float32([[300,image_size[1]],[image_size[0]-300,image_size[1]],[300,0],[image_size[0]-300,0]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    unwarped = cv2.warpPerspective(result_only_Lines, M, image_size, flags=cv2.INTER_LINEAR)
    #Printint curvature in the image
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 1
    color = (0, 0, 0) 
    thickness = 2

    if left_curverad_average == 0:
        img = cv2.putText(img, 'Initializing...', (50,50), font, fontScale, color, thickness, cv2.LINE_AA) 
        img = cv2.putText(img, 'Initializing...', (50,100), font, fontScale, color, thickness, cv2.LINE_AA)
    elif abs(left_curverad_average - right_curverad_average) >  2000 and left_curverad_average <= 5000 and right_curverad_average <= 5000:
        left_line.reset()
        right_line.reset()
    elif (left_curverad_average > 5000 or right_curverad_average > 5000):
        img = cv2.putText(img, 'No curve', (50,50), font, fontScale, color, thickness, cv2.LINE_AA)
        img = cv2.putText(img, 'No curve', (50,100), font, fontScale, color, thickness, cv2.LINE_AA)
    else: 
        img = cv2.putText(img, 'Left curvature = {:.0f}'.format(left_curverad_average) + 'm', (50,50), font, fontScale, color, thickness, cv2.LINE_AA) 
        img = cv2.putText(img, 'Right curvature = {:.0f}'.format(right_curverad_average) + 'm', (50,100), font, fontScale, color, thickness, cv2.LINE_AA) 
    
    if center_avg == 0:
        img = cv2.putText(img, 'Initializing...', (50,150), font, fontScale, color, thickness, cv2.LINE_AA) 
    #elif center_avg > 50:
        #left_line.reset()
    else: 
        img = cv2.putText(img, 'Distance to center = {:.1f}'.format(center_avg) + 'm', (50,150), font, fontScale, color, thickness, cv2.LINE_AA)     
        #img = cv2.putText(img, 'Distance to center = {:.1f}'.format(right_bottom) + 'm', (50,200), font, fontScale, color, thickness, cv2.LINE_AA)     
    result = cv2.addWeighted(img, 1, unwarped, 0.3, 0)
    return result


