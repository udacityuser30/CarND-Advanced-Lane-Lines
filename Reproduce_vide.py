# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:58:33 2020

@author: Sergio Marin
"""
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#from camera_calibration import Camera_Calibration
#from warp_and_threshold import Warp_and_threshold
from main_2 import Main
#from moviepy.editor import VideoFileClip
from camera_calibration import Camera_Calibration

white_output = 'output_video/output_main_7.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(Main) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
