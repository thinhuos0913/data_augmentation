import os
import numpy as np
import cv2
import glob
from PIL import Image

# print(os.getcwd())

# IMAGE PATH
input_folder = './input'
output_folder = './output'

# print(glob.glob(input_folder + "/*.jpg"))

# RESIZE IMAGE
def resize_img(input_img,fx,fy):
	for filename in os.listdir(input_img):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			img_path = os.path.join(input_img, filename)
			img = cv2.imread(img_path)
			resized = cv2.resize(img, None, fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
			new_filename = "resized_" + filename
			new_path = os.path.join(input_img,new_filename)
			cv2.imwrite(new_path,resized)

# resize_img(input_folder,0.5,0.5)

# CROPPING
def crop_img(input_path,save_path,y1,y2,x1,x2):
	for filename in os.listdir(input_path):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			img_path = os.path.join(input_path, filename)
			img = cv2.imread(img_path)
			cropped = img[y1:y2,x1:x2]
			new_filename = "cropped_" + filename
			new_path = os.path.join(save_path, new_filename)
			cv2.imwrite(new_path,cropped)

# ROTATION
def rotation(input_path,save_path,deg):
	for filename in os.listdir(input_path):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			img_path = os.path.join(input_path, filename)
			img = cv2.imread(img_path)
			h,w = img.shape[:2]
			rotation_matrix = cv2.getRotationMatrix2D((w/2,h/2),deg,1.0)
			rotated_img = cv2.warpAffine(img,rotation_matrix,(w,h))
			new_filename = "rotated_" + filename
			new_path = os.path.join(save_path, new_filename)
			cv2.imwrite(new_path, rotated_img)

# INCREASE BRIGHTNESS
def increase_brightness(input_path,save_path,value):
	# Perform rotation for each img in input folder
	for filename in os.listdir(input_path):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			img_path = os.path.join(input_path, filename)
			img = cv2.imread(img_path)
			bright = np.ones(img.shape , dtype="uint8") * value
			increased = cv2.add(img,bright)
			new_filename = "increased_bright_" + filename
			new_path = os.path.join(save_path, new_filename)
			cv2.imwrite(new_path, increased)

# DECREASE BRIGHTNESS
def decrease_brightness(input_path,save_path,value):
	# Perform rotation for each img in input folder
	for filename in os.listdir(input_path):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			img_path = os.path.join(input_path, filename)
			img = cv2.imread(img_path)
			bright = np.ones(img.shape , dtype="uint8") * value
			decreased = cv2.subtract(img,bright)
			new_filename = "decreased_bright_" + filename
			new_path = os.path.join(save_path, new_filename)
			cv2.imwrite(new_path, decreased)

# FLIPPING
def flip_img(input_path,save_path,value):
	for filename in os.listdir(input_path):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			img_path = os.path.join(input_path, filename)
			img = cv2.imread(img_path)
			flip = cv2.flip(img,value)
			new_filename = "flipped_" + filename
			new_path = os.path.join(save_path, new_filename)
			cv2.imwrite(new_path, flip)

# SHARPEN
def sharpen(input_path,save_path):
	for filename in os.listdir(input_path):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			img_path = os.path.join(input_path, filename)
			img = cv2.imread(img_path)
			kernel = np.array([ [0,-1,0],
								[-1,5,-1],
								[0,-1,0] ])
			sharpened = cv2.filter2D(img,-1,kernel)
			new_filename = "sharpened_" + filename
			new_path = os.path.join(save_path, new_filename)
			cv2.imwrite(new_path, sharpened)

# GAUSSIAN BLUR
def gaussian_blur(input_path,save_path,ksize,blur):
	for filename in os.listdir(input_path):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			img_path = os.path.join(input_path, filename)
			img = cv2.imread(img_path)
			gs_blurred = cv2.GaussianBlur(img,ksize,blur)
			new_filename = "gauss_blurred" + filename
			new_path = os.path.join(save_path, new_filename)
			cv2.imwrite(new_path, gs_blurred)

# AVERAGE BLUR
def avg_blur(input_path,save_path,ksize):
	for filename in os.listdir(input_path):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			img_path = os.path.join(input_path, filename)
			img = cv2.imread(img_path)
			blurred = cv2.blur(img,ksize)
			new_filename = "blurred" + filename
			new_path = os.path.join(save_path, new_filename)
			cv2.imwrite(new_path, blurred)

# EDGE DETECTION
def edge_detect(input_path,save_path):
	for filename in os.listdir(input_path):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			img_path = os.path.join(input_path, filename)
			img = cv2.imread(img_path)
			kernel = np.array([ [-1,-1,-1],
								[-1,8,-1],
								[-1,-1,-1] ])
			edge = cv2.filter2D(img,-1,kernel)
			new_filename = "edge_" + filename
			new_path = os.path.join(save_path, new_filename)
			cv2.imwrite(new_path, edge)

# edge_detect(input_folder,output_folder)

# SATURATION
def img_saturation(input_path,save_path,sat):
	for filename in os.listdir(input_path):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			img_path = os.path.join(input_path, filename)
			img = cv2.imread(img_path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			v = img[:, :, 2]
			v = np.where(v <= 255 - sat, v + sat, 255)
			img[:, :, 2] = v
			saturated = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
			new_filename = "saturated_" + filename
			new_path = os.path.join(save_path, new_filename)
			cv2.imwrite(new_path,saturated)

# HUE
def hue_img(input_path,save_path,sat):
	for filename in os.listdir(input_path):
		if filename.endswith(".jpg") or filename.endswith(".png"):
			img_path = os.path.join(input_path, filename)
			img = cv2.imread(img_path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			v = img[:, :, 2]
			v = np.where(v <= 255 + sat, v - sat, 255)
			img[:, :, 2] = v
			hue = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
			new_filename = "hue_" + filename
			new_path = os.path.join(save_path, new_filename)
			cv2.imwrite(new_path,hue)

# DATA AUGMENTATION
def img_augmentation(input_path,save_path):
	crop_img(input_path,save_path,0,300,0,350) # (y1,y2,x1,x2)=(bottom,top,left,right)
	rotation(input_path,save_path,30)
	increase_brightness(input_path,save_path,50)
	decrease_brightness(input_path,save_path,50)
	flip_img(input_path,save_path,1)
	sharpen(input_path,save_path)
	gaussian_blur(input_path,save_path,(5,5),1.0)
	avg_blur(input_path,save_path,(5,5))
	img_saturation(input_path,save_path,75)
	hue_img(input_path,save_path,25)

img_augmentation(input_folder,output_folder)

