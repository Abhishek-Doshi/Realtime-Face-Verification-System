import cv2
import numpy as np
from PIL import Image

def getFaceCoordinates(img, scaleFactor = 1.3, minNeighbors = 5):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor, minNeighbors)
    return faces

def getFaces(image, scaleFactor = 1.3, minNeighbors = 5):
    images = []
    faces = getFaceCoordinates(image, scaleFactor, minNeighbors)
    for (x,y,w,h) in faces:
        img = image[y:y+h, x:x+w]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((224, 224))
        images.append(img_pil)
    return images