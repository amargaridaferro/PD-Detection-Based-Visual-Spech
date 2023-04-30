import numpy as np
import math

import cv2
import dlib
import face_recognition
from imutils import face_utils
from imutils import resize

from utils import resize_image_to_square


def face_detect_nn(image_, net, min_confidence=0.5):
	"""
	min_confidence = minimum probability accepted for face detection
	"""
	# make a copy of the image, to avoid changing original
	image = image_.copy()

	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	(h, w) = image.shape[:2]
	if (h, w) != (720, 1280):
		print('image size not as expected: %i, %i' %(h, w))

	# convert image to 300x300 - necessary for this net
	image = resize_image_to_square(image)
	(h, w) = image.shape[:2]

	# blobFromImage performs pre-processing: mean-subtraction, normalization, 
	# and channel swapping. It takes as input 
	# (image, scalefactor=1.0, size, mean, swapRB=True)
	# ** mean needs to be in the order R,G,B if swapRB=True
	# the parameters used come from
	# https://github.com/opencv/opencv/tree/master/samples/dnn
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (123.0, 177.0, 104.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	#print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()

	confident_detections = []
	confidences = []
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > min_confidence:
			confident_detections.append(i)
			confidences.append(confidence)

	if len(confident_detections) == 1:		
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		i = confident_detections[0]
		confidence = confidences[i]
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# draw the bounding box of the face along with the associated
		# probability
		#text = "{:.2f}%".format(confidence * 100)
		#y = startY - 10 if startY - 10 > 10 else startY + 10
		#cv2.rectangle(image, (startX, startY), (endX, endY),
		#	(0, 0, 255), 2)
		#cv2.putText(image, text, (startX, y),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	
		return image, [startX, startY, endX, endY]
	else:
		return image, []


def get_bif(image_, num_bands=8, num_rotations=12):
	"""
	Computes bio-inspired features, proposed by Guodong Guo, 
	Guowang Mu, Yun Fu, and Thomas S Huang. "Human age estimation
	using bio-inspiredfeatures, 2009"
	based on:
	https://docs.opencv.org/3.4/dc/d12/classcv_1_1face_1_1BIF.html
	"""
	# make a copy of the image, to avoid changing original
	image = image_.copy()
	
	# resize image to ensure all image have the same bif features dimensions
	image = cv2.resize(image, (100, 100))
	
	# convert image to black and white
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# covert image to CV_32F format, required in BIF calculations
	image = np.float32(image)
	image = image * (1.0/255.0)

	# extract buf features
	bif_extractor = cv2.face.BIF_create(num_bands=num_bands, num_rotations=num_rotations)
	bif_features = bif_extractor.compute(image)

	# squeeze
	bif_features = bif_features.squeeze()
	
	#is it necessary re-convert t CV_8U for display and save?
	return bif_features


def get_embeddings(image_):
	"""
	Extracting embeddings from face recognition.
	https://face-recognition.readthedocs.io/en/latest/readme.html#python-module
	Built on top of dlib's resnet model.
	"""
	image = image_.copy()
	embedding = face_recognition.face_encodings(image)
	
	return embedding


def get_landmarks(image, face_box, predictor):
	"""determine the facial landmarks for the face region of one single image, 
	then convert the facial landmark (x, y)-coordinates to a NumPy
	array."""

	# copy image
	landmarks_img = image.copy()

	# get coordinates of face box
	[startX, startY, endX, endY] = face_box
	
	# convert rectangle to dlib style
	face_rect = dlib.rectangle(startX, startY, endX, endY)

	# predict landmarks
	gray_image = cv2.cvtColor(landmarks_img, cv2.COLOR_BGR2GRAY)
	shape = predictor(gray_image, face_rect)
	shape = face_utils.shape_to_np(shape)
		
	# convert dlib's rectangle to a OpenCV-style bounding box
	(x, y, w, h) = face_utils.rect_to_bb(face_rect)
	#cv2.rectangle(landmarks_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
	# show the face number
	#cv2.putText(landmarks_img, "Face", (x - 10, y - 10),
	#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for x, y in shape:
		cv2.circle(landmarks_img, (x, y), 1, (0, 0, 255), -1)

	return landmarks_img, shape


def get_facial_features(shape_, image_name, remove_non_frontal_frames=True):

	# make a copy of the shape, to avoid changing original
	shape = shape_.copy()
	
	tl = shape[2-1] # the most left point in the face
	tr = shape[16-1] # the most right point
	exl = shape[37-1] # eye exterior left
	enl = shape[40-1] # eye interior left
	enr = shape[43-1] # eye exterior right
	exr = shape[46-1] # eye interior right
	sto = shape[67-1] #center, inferior lip
	gn = shape[9-1] # end chin
	gol = shape[5-1] #
	n = shape[28-1] # between the eyes

	eye_2 = shape[38-1]
	eye_3 = shape[39-1]
	eye_5 = shape[41-1]
	eye_6 = shape[42-1]

	face_eyelevel_left = shape[1-1]
	face_eyelevel_right = shape[17-1]
	face_mouthlevel_left = shape[4-1]
	face_mouthlevel_right = shape[14-1]

	mouth_left = shape[49-1]
	mouth_right = shape[55-1]

	if remove_non_frontal_frames:
		if not is_frontal(
			exl, exr, mouth_left, mouth_right, face_eyelevel_left,
			face_eyelevel_right, face_mouthlevel_left, face_mouthlevel_right,
			tolerance=0.1):
			return []

	face_width = distance(tl, tr)
	eye_width = distance(exl, enl)
	binocular_width = distance(exl, exr)
	mandibular_length = distance(sto, gn)
	cranial_base_area = (face_width + binocular_width) * (tl[1] - exl[1]) /2 #(B+b)*h/2
	mandibular_nasation_angle = getAngle(gn, n, gol)
	
	if (eye_width == 0) or (eye_area == 0):
			return []
	
	# eye area for scale:
	(x, y) = zip(exl,eye_2,eye_3,enl,eye_5,eye_6)
	eye_area = PolyArea(x, y)	

	feats = [face_width/eye_width, binocular_width/eye_width, mandibular_length/eye_width, cranial_base_area/eye_area, mandibular_nasation_angle]
	
	if np.isnan(np.sum(feats)) or np.isinf(np.sum(feats)):
		return []
	else:
		return feats

def discard_outliers(features, ids, imgs=[], bifs=[], emb=[]):
	"""
	Computes IQR. Discards points:
		> Q3 + 1.5 IQR
		< Q1 - 1.5 IQR
	"""
	Q1 = np.quantile(features, 0.25, axis=0)
	Q3 = np.quantile(features, 0.75, axis=0)
	IQR = Q3 - Q1

	assert len(features) == len(ids), "error in discard outliers, dimensions do not match"
	not_outlier = (features > (Q1 - 1.5 * IQR)) & (features < (Q3 + 1.5 * IQR))
	
	features = np.array(features)
	ids = np.array(ids)

	features_no_outliers = features[not_outlier.all(axis=1)]
	ids_no_outliers = ids[not_outlier.all(axis=1)]
	if len(imgs):
		imgs = np.array(imgs)
		imgs = imgs[not_outlier.all(axis=1)].tolist()
	if len(bifs):
		bifs = np.array(bifs)
		bifs = bifs[not_outlier.all(axis=1)].tolist()
	if len(emb):
		emb = np.array(emb)
		emb = emb[not_outlier.all(axis=1)].tolist()

	return features_no_outliers, ids_no_outliers.tolist(), imgs, bifs, emb

def is_frontal(exl, exr, mouth_l, mouth_r, face_eyelevel_l,
		face_eyelevel_r, face_mouthlevel_l, face_mouthlevel_r,
		tolerance=0.4):
	"""
	compares distance of each eye to face margin and of each mouth extremity to
	corresponding face margin. 
	Considers frontal position if ratios between distances are larger than the
	tolerance and smaller than one.

	Returns True if Face is frontal. 
	"""

	l_eye_face_distance = distance(exl, face_eyelevel_l)
	r_eye_face_distance = distance(exr, face_eyelevel_r)
	
	l_mouth_face_distance = distance(mouth_l, face_mouthlevel_l)
	r_mouth_face_distance = distance(mouth_r, face_mouthlevel_r)
	
	if (l_eye_face_distance ==0 or r_eye_face_distance==0 or l_mouth_face_distance==0 or r_mouth_face_distance==0):
			return []
	
	# either both ratios are 1 or one is > 1 and the other is < 1. We pick the smaller
	a = min(l_eye_face_distance/l_eye_face_distance, r_eye_face_distance/r_eye_face_distance)
	b = min(l_mouth_face_distance/l_mouth_face_distance, l_mouth_face_distance/r_mouth_face_distance)

	return ((a > 1 - tolerance) and (b > 1 - tolerance))
	

def distance(a, b):
	return math.sqrt( (b[0] - a[0])**2 + (b[1] - a[1])**2 )

def getAngle(a, b, c):
	ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
	return ang + 360 if ang < 0 else ang

def PolyArea(x,y):
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
