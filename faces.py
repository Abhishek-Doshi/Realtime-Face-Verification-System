import cv2
import numpy as np
from PIL import Image

def getFaceCoordinates(image, scaleFactor = 1.3, minNeighbors = 5):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor, minNeighbors)
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

def highlightFaces(image, scaleFactor = 1.3, minNeighbors = 5):
    faces = getFaceCoordinates(image, scaleFactor, minNeighbors)
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 1)
    return image

def viewImage(name, highlight_Faces = False):
    image = cv2.imread(name)
    window_name = name
    if highlight_Faces:
        image = highlightFaces(image)
    cv2.imshow(window_name, image) 
    cv2.waitKey(0)