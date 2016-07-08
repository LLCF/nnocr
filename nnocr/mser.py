'''
Returns MSERs for an image

'''
import cv2
import numpy as np

def get_mser(img):

	# check if image is grayscale

	if len(img.shape) == 3:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	elif len(img.shape) == 1:
		gray = img
	else:
		raise ValueError('Invalid shape for image in get_mser.py, got image with shape %s'%str(img_shape))

	(rows,cols) = gray.shape
	num_pixels = rows * cols

	# valid mser must have num pixels > 0.01% of image and < 40% of image
	min_px = int((0.01/100.0)*num_pixels)
	max_px = int(0.4*num_pixels)

	# delta is set to 1 to maximise recall
	mser = cv2.MSER_create(_delta = 1, _min_area = min_px, _max_area = max_px)

	(regions,bboxes) = mser.detectRegions(gray)

	return (regions,bboxes)
