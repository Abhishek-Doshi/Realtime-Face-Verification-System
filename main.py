from test_model import verifyFace, getFeatureVector
from model_images import getFaces
import numpy as np
import cv2
from PIL import Image
import time

while True:

	tic = time.time()

	IMG_1 = cv2.imread('1.jpg')
	IMG_2 = cv2.imread('2.jpg')

	faces_in_1 = getFaces(IMG_1)
	faces_in_2 = getFaces(IMG_2)
	face_1 = None
	face_2 = None

	if ((len(faces_in_2)*len(faces_in_1)) != 0):
		face_1 = faces_in_1[0]
		face_2 = faces_in_2[0]

	if ((face_1 != None) & (face_2 != None)):
		if verifyFace(face_1, face_2, print_score = True): 
			toc = time.time()
			time_elapsed = toc - tic
			print('Face Verified! ', end = '')
		else: 
			toc = time.time()
			time_elapsed = toc - tic
			print('Face Not Verified! ', end = '')
		print('Time Required - ' + str(time_elapsed) + '\n')
	
	else: print('No Face Detected.\n')