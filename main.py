from model import verifyFace
from faces import getFaces
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

	face_1 = faces_in_1[0]
	face_2 = faces_in_2[0]

	toc = time.time()
	time_elapsed = toc - tic

	if verifyFace(face_1, face_2):
		print('Face Verified! ', end = '')
	else: print('Face Not verified! ', end = '')
	print('Time Required - ', time_elapsed)