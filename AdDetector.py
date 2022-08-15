# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:16:58 2019

@author: willi
"""
# What I want to do is take four points, analyze the general size in pixels 
# area. So if I get a 728x90, then I can that those four

import os
from os.path import join
import cv2
import numpy as np
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
import sys

deprecation._PRINT_DEPRECATION_WARNINGS = False

# turn off excessive logging
tf.logging.set_verbosity(tf.logging.ERROR)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

import time

def shape_detector(xmin, xmax, ymin, ymax):
    xlength = xmax - xmin
    ylength = ymax - ymin
    
    if 0.8 < xlength / ylength < 1.2:
        print("Shape is likely a Medium Rectangle / 300x250")
        return '300x250'
    
    elif 6 < xlength / ylength < 10:
        print("Shape is likely a Billboard / 728x90")
        return '728x90'
    
    elif 0.4 < xlength / ylength < 0.6:
        print("Shape is likely a Half-Page / 300x600")
        return '300x600'
    
    elif 0.2 < xlength / ylength < 0.3:
        print("Shape is likely a Skyscraper / 160x600")
        return '160x600'
    
    else:
        print("Not sure what size this is.")
        return "Unknown"
    

def midpoint(xmin, xmax, ymin, ymax):
    x_mid = ((xmin+xmax) / 2)
    y_mid = ((ymin+ymax) / 2)
    return (int(x_mid), int(y_mid))

# Load the Tensorflow model into memory.
def get_ad_location():
    MODEL_NAME = 'inference_graph'
    Image_directory='dataset'
    output_folder_path = os.path.join(os.getcwd(), 'output_folder')
    
    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    
    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
    PATH_To_dataset=os.path.join(CWD_PATH,Image_directory)#dataset path
    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
    
    # Path to image
    
    # Number of classes the object detector can identify
    NUM_CLASSES = 1
    
    # Load the label map.
    # Label maps map indices to category names
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
    
	# Define input and output tensors (i.e. data) for the object detection classifier

	# Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

	# Output tensors are the detection boxes, scores, and classes
	# Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

	# Each score represents level of confidence for each of the objects.
	# The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

	# Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    innerimages = os.listdir(PATH_To_dataset)
	# Load image using OpenCV and
	# expand image dimensions to have shape: [1, None, None, 3]
	# i.e. a single-column array, where each item in the column has the pixel RGB value
    counter_write=len(os.listdir(output_folder_path))
    # For one-time, single use image ad detection
    unix_timestamp = int(datetime.timestamp(datetime.now()))
    # pyautogui.screenshot(os.path.join(PATH_To_dataset, 'screenshot.jpg'))
    
    for img in innerimages:
        image = cv2.imread(join(PATH_To_dataset, str(img)))
        image_expanded = np.expand_dims(image, axis=0)

		# Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})
        #add this part to count objects
        final_score = np.squeeze(scores)    
        count = 0
        for i in range(100):
            if scores is None or final_score[i] > 0.8:
                count = count + 1        
        mid_point_arr=[]
        
        # points is where the array of each box is kept
        points=[]
        
        im_height, im_width = image.shape[:2]
		#Detected xmin,ymin,xmax,ymax and mid points
        for i in range(0,count):
            position = boxes[0][i]
            (xmin, xmax, ymin, ymax) = (position[1]*im_width, 
                                        position[3]*im_width, 
                                        position[0]*im_height, 
                                        position[2]*im_height)
            
            #Formula to find mid points
            val=[xmin, xmax, ymin, ymax]
            mid_points=(xmax-xmin)+(ymax-ymin)
            mid_point_arr.append(int(mid_points))
            points.append(val)
            
        vis_util.visualize_boxes_and_labels_on_image_array(image,
                                                           np.squeeze(boxes),
                                                           np.squeeze(classes).astype(np.int32),
                                                           np.squeeze(scores),
                                                           category_index,
                                                           use_normalized_coordinates=True,
                                                           line_thickness=8,
                                                           min_score_thresh=0.8)

		# All the results have been drawn on image. Now display the image.
        cv2.imwrite(f"output_folder\Detected_image_{unix_timestamp}_"+str(img)+".png",image)
        
        mid_point_container = []
        
        if len(points) == 0:
            print("no ads detected on page")
            result = 0
        else:
            print(f"points: {points}")
            
            for point in points:
                xmin, xmax, ymin, ymax = point
                shape = shape_detector(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                print("shape: {}".format(shape)) # for debugging
                mid_point = midpoint(xmin, xmax, ymin, ymax)
                print(f"Located at midpoint: {mid_point}")
                mid_point_container.append(mid_point)
            
            print(mid_point_container)
            result = mid_point_container
            
        write_file_string = "Detected_image_{}_{}".format(unix_timestamp, str(img))
        with open('data_sheet.txt', 'a') as file:
            content = str(counter_write) + "\t" + str(unix_timestamp) + "\t" + str(write_file_string) + "\t" + str(len(points)) + "\t" + str(result)
            file.write(content+"\n")
        counter_write=counter_write+1            
            
            
        
if __name__ == "__main__":
    time.sleep(2)
    ad_location = get_ad_location()
    if type(ad_location) is str:
        print("No ads on page!")
    else:
        print(f"Ad location: {ad_location}")
        # pyautogui.moveTo(ad_location)