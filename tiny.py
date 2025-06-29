import tensorflow as tf
import numpy as np
from numpy import expand_dims
from keras.models import load_model, Model
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
 
from tensorflow.keras import backend
from tensorflow.python.keras import backend as K

import cv2
import time
import os
import socket
from io import BytesIO



config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes
		self.label = -1
		self.score = -1

	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)

		return self.label

	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]

		return self.score

def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	nb_class = netout.shape[-1] - 5
	boxes = []
	
	# Apply sigmoid to box x, y coordinates and class probabilities
	netout[..., :2]  = _sigmoid(netout[..., :2])
	netout[..., 4:]  = _sigmoid(netout[..., 4:])
	
	# Calculate class scores and apply object threshold
	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
	netout[..., 5:] *= netout[..., 5:] > obj_thresh
	
	# Loop through each grid cell and anchor box
	for row in range(grid_h):
		for col in range(grid_w):
			for b in range(nb_box):
				# Objectness score
				objectness = netout[row, col, b, 4]
				if objectness <= obj_thresh:  # Skip low-confidence boxes earlier
					continue

				# Bounding box coordinates (x, y, w, h)
				x, y, w, h = netout[row, col, b, :4]
				x = (col + x) / grid_w  # center position, unit: image width
				y = (row + y) / grid_h  # center position, unit: image height
				w = anchors[2 * b + 0] * np.exp(w) / net_w  # width in image units
				h = anchors[2 * b + 1] * np.exp(h) / net_h  # height in image units

				# Class probabilities
				classes = netout[row, col, b, 5:]
				
				# Create a bounding box and append it if the objectness score is high
				box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
				#print(box.xmin, box.ymin, box.xmax, box.ymax)
				boxes.append(box)
	return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
	new_w, new_h = net_w, net_h
	for i in range(len(boxes)):
		x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
		y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
		boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
		boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
		boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
		boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b
	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
			return 0
		else:
			return min(x2,x4) - x3

def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
	intersect = intersect_w * intersect_h
	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	union = w1*h1 + w2*h2 - intersect
	return float(intersect) / union

def do_nms(boxes, nms_thresh):
	if len(boxes) > 0:
		nb_class = len(boxes[0].classes)
	else:
		return
	for c in range(nb_class):
		sorted_indices = np.argsort([-box.classes[c] for box in boxes])
		for i in range(len(sorted_indices)):
			index_i = sorted_indices[i]
			if boxes[index_i].classes[c] == 0: continue
			for j in range(i+1, len(sorted_indices)):
				index_j = sorted_indices[j]
				if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
					boxes[index_j].classes[c] = 0


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
	v_boxes, v_labels, v_scores = list(), list(), list()
	
	# enumerate all boxes
	for box in boxes:
		# enumerate all possible labels
		for i in range(len(labels)):
			# check if the threshold for this label is high enough
			if box.classes[i] > thresh:
				v_boxes.append(box)
				v_labels.append(labels[i])
				v_scores.append(box.classes[i]*100)
				# don't break, many labels may trigger for one box
	return v_boxes, v_labels, v_scores

def load_image_pixels(image):
	# load the image to get its shape

	image = img_to_array(image)
	# scale pixel values to [0, 1]
	image = image.astype('float32')
	image /= 255.0
	# add a dimension so that we have one sample
	image = expand_dims(image, 0)
	return image



model = load_model("YOLOv3-tiny.h5", compile=False)

cam_port = 0
cam = cv2.VideoCapture(cam_port)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)

anchors = [[81,82,  135,169,  344,319], [10,14,  23,27,  37,58]] #YOLOv3-tiny Anchors

# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


# define the probability threshold for detected objects
class_threshold = 0.40

iterations = 0
timeArray = []

FPSTimer1 = time.time()

while(1):
	
	# define our new photo
	#start_time = time.time()
	# reading the input using the camera 
	time1 = time.time()
	result, img = cam.read()
	""" cv2.imwrite('temp.jpg', img) """
	""" cv2.waitKey(1)
	cv2.imshow('Webcam Feed', img) """

	#_, buffer = cv2.imencode('.jpg', img)


	# Load and prepare image
	image = load_image_pixels(img)


	#Get the intermediate result
	
	result = model(image)

	result[0] = result[0].numpy()
	result[1] = result[1].numpy()

	boxes = list()

	for i in range(len(result)):
		#print(i)
		# decode the output of the network
		boxes += decode_netout(result[i][0], anchors[i], class_threshold, 416, 416)

	#print(boxes)

	

	# correct the sizes of the bounding boxes for the shape of the image
	correct_yolo_boxes(boxes, 416, 416, 416, 416)
	# suppress non-maximal boxes
	#print(boxes)
	do_nms(boxes, 0.5)

	# get the details of the detected objects
	v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
	
	image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	#image = np.zeros((416, 416, 3), dtype=np.uint8)

	itemList = []

	for i in range(len(v_boxes)):
		box, label, score = v_boxes[i], v_labels[i], v_scores[i]
		print(label,score)
		y1, x1, y2, x2 = box.ymin - 75, box.xmin - 20, box.ymax - 50, box.xmax - 20
		#print(x1,y1,x2,y2,label, score)

		if label in itemList:
			continue
		else:
			itemList.append(label)
		
		cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

		# Set the position for the label, which will be at the top-left corner of the rectangle
		label_x = x1
		label_y = y1 - 10  # Adjust this if you want the label inside or slightly above the rectangle

		(text_width, text_height), baseline = cv2.getTextSize(v_labels[i] + "  " + str(v_scores[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

		# Draw a filled rectangle for the text background
		cv2.rectangle(image, (label_x, label_y - text_height - baseline), 
					(label_x + text_width, label_y + baseline), 
					(0, 255, 0), cv2.FILLED)

		# Put the label text above the rectangle
		cv2.putText(image, v_labels[i] + "  " + str(v_scores[i]), (label_x, label_y - baseline), 
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
	
	image = cv2.resize(image, (832, 832))
	# Display the image
	cv2.imshow('Image with Box', image)
	cv2.waitKey(1)
	#cv2.destroyAllWindows()


	""" end_time = time.time()
	execution_time = end_time - start_time
	print(execution_time) """

	iterations += 1

	elapsed_time = time.time() - FPSTimer1
	if elapsed_time >= 1:
		print(f"FPS in the last second: {iterations}")
		
		# Reset the timer and iterations counter
		FPSTimer1 = time.time()
		iterations = 0

""" 	time2 = time.time()
	timeArray.append(time2-time1)
	iterations += 1

	if iterations >= 100:
		iterations = 0
		print(sum(timeArray)/len(timeArray))
		timeArray.clear() """