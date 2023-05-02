#April 2023, based on Catarina Botelho work
#Ana Margarida Ferro
import os
import argparse
import numpy as np
np.random.seed(124)
import cv2


def arg_parser():
		
	parser = argparse.ArgumentParser()

	parser.add_argument('-input_dir', 
		default='/ffs/tmp/mctb/projects/osa_images/data/vlog_frames/c_001/')
	parser.add_argument('-faces_dir', 
		default='/ffs/tmp/mctb/projects/osa_images/data/vlog_main_face/c_001/')
	parser.add_argument('-landmarks_dir', 
		default='/ffs/tmp/mctb/projects/osa_images/data/vlog_face_landmark/c_001/')
	parser.add_argument('-features_dir', 
		default='/ffs/tmp/mctb/projects/osa_images/data/features/c_001/')
	parser.add_argument('-bifs_dir', 
		default='/ffs/tmp/mctb/projects/osa_images/data/bifs/c_001/')
	parser.add_argument('-embeddings_dir', 
		default='/ffs/tmp/mctb/projects/osa_images/data/embeddings/c_001/')
	parser.add_argument('-image_count_file', 
		default='/ffs/tmp/mctb/projects/osa_images/data/image_count.csv')
	parser.add_argument('-corrected_faces_dir', 
		default='/ffs/tmp/mctb/projects/osa_images/data/vlog_main_face_corrected/c_001/')
	parser.add_argument('-no_backg_faces_dir', 
		default='/ffs/tmp/mctb/projects/osa_images/data/vlog_main_face_no_backg/c_001/')

	parser.add_argument('-compute_landmarks',
		dest='compute_landmarks',
		action='store_true')
	parser.add_argument('-compute_facial_feats',
		dest='compute_facial_feats',
		action='store_true')
	parser.add_argument('-compute_bifs',
		dest='compute_bifs',
		action='store_true')
	parser.add_argument('-compute_face_embeddings',
		dest='compute_face_embeddings',
		action='store_true')
	parser.add_argument('-exclude_non_frontal',
		dest='exclude_non_frontal',
		action='store_true')
	parser.add_argument('-exclude_outliers',
		dest='exclude_outliers',
		action='store_true')
			
	parser.add_argument('-save_faces',
		dest='save_faces',
		action='store_true')
	parser.add_argument('-save_faces_w_landmarks',
		dest='save_faces_w_landmarks',
		action='store_true')
	parser.add_argument('-save_features',
		dest='save_features',
		action='store_true')

	opt = parser.parse_args()

	return opt


def read_images(path, size=None):
	"""
	example: images, img_list = read_images('path/to/image/dir', size=(70,70))
	"""
	image_list = os.listdir(path)
	image_list.sort()
	
	images = np.empty(len(image_list), dtype=object)
	for n in range(len(image_list)):
		images[n] = cv2.imread(os.path.join(path, image_list[n]))

	if (size is not None):
		for n in range(len(image_list)):
			images[n]  = cv2.resize(images[n], size)
	
	return images, image_list


def save_images(images, image_names, output_dir):
	for img, name in zip(images, image_names):
		cv2.imwrite(output_dir + "/" + name, img)


def resize_image_to_square(image_, dim=300, crop=False):
	
	# make a copy of the image, to avoid changing original
	image = image_.copy()
	
	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	(h, w, channels) = image.shape

	if crop:
		# crop image dim to square
		if w > h: 
			desired_start_w = int((w-h)/2)
			image = image[:, desired_start_w:desired_start_w+h]
		elif w<h:
			desired_start_h = int((h-w)/2)
			image = image[desired_start_h:desired_start_h+w, :]
	else:
		# pad image to square
		if w > h:
			image = cv2.copyMakeBorder(
				src=image, top=(w-h), bottom=0, left=0, right=0, 
				borderType=cv2.BORDER_CONSTANT, value=0)
			#new_image = np.zeros((w, w, channels))
			#new_image[:h] = image
		elif w<h:
			image = cv2.copyMakeBorder(
				src=image, top=0, bottom=0, left=0, right=(h-w), 
				borderType=cv2.BORDER_CONSTANT, value=0)
			#new_image = np.zeros((h, h, channels))
			#new_image[:, :w] = image
	
	# 2nd: resize to 300x300
	image = cv2.resize(image, (300, 300))
	
	return image

def crop_to_box(image_, box, margin=1, square=True):
	"""
	crops image around the box + box_dimension*margin.
	margin = 1 gives a margin as large as the box itself.
	square = True coverts the box to a square, using the
	max of the boxes dimension.
	"""
	# make a copy of the image, to avoid changing original
	image = image_.copy()
	
	(h, w, channels) = image.shape
	[startX, startY, endX, endY] = box
	x_len = endX - startX
	y_len = endY - startY

	if square:
		x_len = max(x_len, y_len)
		y_len = max(x_len, y_len)
		endX = startX + x_len if startX + x_len <= w else endX
		endY = startY + y_len if startY + y_len <= h else endY

	# compute margins for each side of the box
	x_margin = int(x_len * margin /2)
	y_margin = int(y_len * margin /2)

	# define croping points:
	new_startX = startX - x_margin if startX - x_margin >= 0 else 0
	new_endX = endX + x_margin if endX + x_margin <= w else w
	new_startY = startY - y_margin if startY - y_margin >= 0 else 0
	new_endY = endY + y_margin if endY + y_margin <= h else h

	# if square, check if both dimensions are equal
	if square and ((new_endY - new_startY) != (new_endX - new_startX)):
		smaller_len = min((new_endY - new_startY), (new_endX - new_startX))
		new_endX = new_startX + smaller_len
		new_endY = new_startY + smaller_len
	
	# crop image:
	image = image[new_startY:new_endY, new_startX:new_endX]

	return image



def correct_illumination(images):
	"""
	performs OpenCV's CLAHE (Contrast Limited Adaptive Histogram Equalization)
	adapted from https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c
	"""
	final_images = []
	for img in images:
		# Converting image to LAB Color model
		lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

		# Splitting the LAB image to different channels
		l, a, b = cv2.split(lab)

		# Applying CLAHE to L-channel 
		clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
		cl = clahe.apply(l)

		# Merge the CLAHE enhanced L-channel with the a and b channel
		l_img = cv2.merge((cl,a,b))

		# Converting image from LAB Color model to RGB model
		final = cv2.cvtColor(l_img, cv2.COLOR_LAB2BGR)
		
		final_images.append(final)
	return final_images


def remove_background(images):
	"""
	Removes background of image, and paits it black
	adapted from https://stackoverflow.com/questions/31133903/opencv-remove-background
	"""
	final_images = []
	for img in images:
		# Image dims
		height, width = img.shape[:2]

		# Create a mask holder
		mask = np.zeros(img.shape[:2],np.uint8)

		# Grab Cut the object
		bgdModel = np.zeros((1,65),np.float64)
		fgdModel = np.zeros((1,65),np.float64)

		# Hard Coding the Rect The object must lie within this rect.
		rect = (10,10,width-30,height-30)
		cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
		mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
		img_final = img*mask[:,:,np.newaxis]

		final_images.append(img_final)
	return final_images
		
