import numpy as np
import cv2
import sys
import math
import pdb

import model
import scipy.spatial.distance
import networkx as nx

def make_lines_from_pair(pairs,bboxes,bbox_class,bbox_prob):
	
	G = nx.Graph()

	for pair in pairs:
		G.add_edge(pair[0],pair[1])
		print(pair)

	subgraphs = list(nx.connected_component_subgraphs(G))

	for subgraph in subgraphs:
		print('Subgraph: ')
		cxs = []
		node_list = subgraph.nodes()
		for node in node_list:
			(x,y,w,h) = bboxes[node]
			cx = x + int(w/2) 
			
			#print(bbox_class[node])
			cxs.append(cx)
		cxs = np.array(cxs)
		line_order = np.argsort(cxs)
		for i in line_order:
			node = node_list[i]
			print(bbox_class[node])
		#pdb.set_trace()


	return 0


def find_pairs(img,regions,bboxes,bbox_class,bbox_prob):

	pairs = []
	mean_colors = []

	#get mean colors
	for idx,region in enumerate(regions):
		color_vals = img[region[:,1],region[:,0]]
		mean_color = np.mean(color_vals,axis = 0)
		mean_colors.append(mean_color)
		#print(mean_color)
		#vis = img.copy()
		#bbox = bboxes[idx]
		#cv2.rectangle(vis,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(127,127,0),1)
		#cv2.imshow('img',vis)
		#cv2.waitKey(0)

	for idx1,bbox1 in enumerate(bboxes):
		if bbox_class[idx1] == '-1':
			continue
		(x1,y1,w1,h1) = bbox1
		#pdb.set_trace()
		for idx2,bbox2 in enumerate(bboxes):
			if idx2 <= idx1:
				continue
			if bbox_class[idx2] == '-1':
				continue

			(x2,y2,w2,h2) = bbox2

			# check if nested regions
			(ux,uy,uw,uh) = intersection(bbox1,bbox2)
			area_intersect = uw * uh
			area_1 = w1 * h1
			area_2  = w2 * h2
			if area_intersect > 0.9 * area_1 or area_intersect > 0.9 * area_2:
				continue

			# check height ratio
			height_ratio = h2/float(h1)
			if height_ratio > 2.0 or height_ratio < 0.5:
				continue
			# check distance
			dx = x1-x2
			dy = y1-y2
			dist = math.sqrt(dx*dx+dy*dy)

			if dist > 3.0 * max(w1,w2):
				continue

			# check if on same line
			cy1 = y1 + h1/2.0
			cy2 = y2 + h2/2.0
			if cy1 < y2 or cy1 > y2 + h2 or cy2 < y1 or cy2 > y1 + h1:
				continue

			# check color
			col1 = mean_colors[idx1]
			col2 = mean_colors[idx2]
			if scipy.spatial.distance.euclidean(col1,col2) > 30:
				#pass
				continue

			pairs.append((idx1,idx2))

	return pairs

def union(a,b):
	x = min(a[0], b[0])
	y = min(a[1], b[1])
	w = max(a[0]+a[2], b[0]+b[2]) - x
	h = max(a[1]+a[3], b[1]+b[3]) - y
	return (x, y, w, h)

def intersection(a,b):
	x = max(a[0], b[0])
	y = max(a[1], b[1])
	w = min(a[0]+a[2], b[0]+b[2]) - x
	h = min(a[1]+a[3], b[1]+b[3]) - y
	if w<0 or h<0:
		return (0,0,0,0)
	return (x, y, w, h)

def calc_rect_iou(rect1,rect2):

	intersection1 = intersection(rect1,rect2)
	union1 = union(rect1,rect2)

	area_intersect = intersection1[2] * intersection1[3]
	area_union = union1[2] * union1[3]

	iou = 0

	if area_intersect > 0:
		iou = area_intersect / float(area_union)

	return iou


def format_img(img,img_size):

	img = cv2.resize(img,(img_size,img_size))

	# image was normalised in training, so we normalise here
	img = img/255.0
	img -= 0.5
	
	img = np.expand_dims(img,axis = 0)

	img = img.astype('float32')

	return img


def get_chars(img,char_model,regions,bboxes):


	if len(img.shape) == 3:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	elif len(img.shape) == 1:
		gray = img
	else:
		raise ValueError('Invalid shape for image in nn.py, got image with shape %s'%str(img_shape))

	vis = img.copy()

	vis_probs = True
	if vis_probs:
		prob_map = 0 * gray.copy()


	img_size = 24

	(rows,cols) = gray.shape

	threshold = 0.5

	num_char_regions = 0

	num_regions = len(bboxes)

	bbox_class = ['-1'] * num_regions
	bbox_prob = [0.0] * num_regions

	for idx,bbox in enumerate(bboxes):

		sub_imgs = []

		x = bbox[0]
		y = bbox[1]
		w = bbox[2]
		h = bbox[3] 

		aspect_ratio = h / float(w)
		if aspect_ratio > 10 or aspect_ratio < 0.1:
			continue
		# if height is greater than width, widen the box to match the height
		# helps match the training data
		'''
		if h > w:
			diff = h - w
			x -= int(diff/2)
			w = h
		'''
		# take three crops of the character
		for seg_idx in xrange(1,2):

			x1 = x - seg_idx * 3
			y1 = y - seg_idx * 3
			w1 = w + seg_idx * 3
			h1 = h + seg_idx * 3

			if x1 < 0:
				x1 = 0
			if y1 < 0:
				y1 = 0
			if x1 + w1 >= cols:
				w1 = cols - x1
			if y1 + h >= rows:
				h1 = rows - y1

			sub_img = gray[y1:y1+h1,x1:x1+w1]

			sub_img_k = format_img(sub_img,img_size)
			sub_imgs.append(sub_img_k)

		sub_imgs = np.array(sub_imgs)

		# get the predicted character for the box
		pred_char = char_model.predict(sub_imgs,batch_size=1)
		pred_char = np.mean(pred_char,axis=0)
		pred_char_sorted = np.argsort(pred_char)
		char_max = pred_char_sorted[-1]

		char_prob = pred_char[char_max]

		detected_char = model.char_list[char_max]

		if detected_char == '0' or detected_char == 'O':
			second_char_max = pred_char_sorted[-2]
			second_char_prob = pred_char[second_char_max]
			second_detected_char = model.char_list[second_char_max]
			if second_detected_char == '0' or second_detected_char == 'O':
				char_prob = char_prob + second_char_prob
		'''
		if detected_char == 'I': #or detected_char == 'O':
			argsort = np.argsort(pred_char)
			loc1 = argsort[-1]
			loc2 = argsort[-2]
			char1 = model.char_list[loc1]
			char2 = model.char_list[loc2]
			prob1 = pred_char[loc1]
			prob2 = pred_char[loc2]

			print('Det 1 -> ' + str(prob1) + ' | ' + char1)
			print('Det 2 -> ' + str(prob2) + ' | ' + char2)
			cv2.imshow('s',sub_img)
			cv2.waitKey(0)
		'''
		
		if vis_probs:
			normalised_prob = int(100*char_prob)
			vis_vals = prob_map[regions[idx][:,1],regions[idx][:,0]]
			vis_locs = vis_vals < normalised_prob
			vis_locs = np.expand_dims(vis_locs,axis=1)
			vis_locs = np.repeat(vis_locs,2,axis=1)

			lower_prob_locs = np.extract(vis_locs,regions[idx])
			lower_prob_locs = lower_prob_locs.reshape((-1,2))

			prob_map[lower_prob_locs[:,1],lower_prob_locs[:,0]] = normalised_prob


		if char_prob > threshold:
			bbox_class[idx] = detected_char
			bbox_prob[idx] = char_prob
			num_char_regions += 1
	

	for ix,bbox1 in enumerate(bboxes):
		if bbox_class[ix] == '-1':
			continue

		for jy,bbox2 in enumerate(bboxes):
			if jy <= ix:
				continue
			if bbox_class[jy] == '-1':
				continue
			iou = calc_rect_iou(bbox1,bbox2)
			if iou > 0.95 or (iou > 0.9 and bbox_class[ix] == bbox_class[jy]):
				if bbox_prob[ix] > bbox_prob[jy]:
					bbox_prob[jy] = 0.0
					bbox_class[jy] = '-1'
				else:
					bbox_prob[ix] = 0.0
					bbox_class[ix] = '-1'
					break

	pairs =	find_pairs(img,regions,bboxes,bbox_class,bbox_prob)
	#all_vals_in_pairs = [item for sublist in pairs for item in sublist]

	vis = img.copy()
	
	for pair in pairs:
		vis = img.copy()
		ix = pair[0]
		jy = pair[1]
		#print(str((ix,jy)))
		bbox1 = bboxes[ix]
		bbox2 = bboxes[jy]

		cv2.rectangle(vis,(bbox1[0],bbox1[1]),(bbox1[0]+bbox1[2],bbox1[1]+bbox1[3]),(127,127,0),1)
		cv2.putText(vis,bbox_class[ix] ,(bbox1[0],bbox1[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,50,50),2)

		cv2.rectangle(vis,(bbox2[0],bbox2[1]),(bbox2[0]+bbox2[2],bbox2[1]+bbox2[3]),(127,127,0),1)
		cv2.putText(vis,bbox_class[jy] ,(bbox2[0],bbox2[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,50,50),2)

		#cv2.imshow('img', vis)
		#cv2.waitKey(0)
	
	for ix,bbox in enumerate(bboxes):
		if bbox_prob[ix] > threshold:
			in_pair = False
			for pair in pairs:
				if ix == pair[0] or ix == pair[1]:
					in_pair = True
					break
			if in_pair:
				cv2.rectangle(vis,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(127,127,0),1)
				cv2.putText(vis,bbox_class[ix] ,(bbox[0],bbox[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,50,50),2)
				#cv2.putText(vis,str(bbox[3]/float(bbox[2])) ,(bbox[0],bbox[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,50,50),2)
				# bbox[3] is height, bbox[2] is width

	make_lines_from_pair(pairs,bboxes,bbox_class,bbox_prob)

	prob_map = prob_map * 2

	vis_jet = cv2.applyColorMap(prob_map,cv2.COLORMAP_JET)

	cv2.imshow('a',vis_jet)
	cv2.imshow('img', vis)
	cv2.waitKey(0)

	print('Num valid regions = %d' % num_char_regions)

	return 0
