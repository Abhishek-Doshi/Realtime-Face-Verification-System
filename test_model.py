import tensorflow as tf
import keras
import model_images
from keras import backend as K
from keras import applications
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.models import model_from_json, Model, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.models import model_from_json
from PIL import Image
import numpy as np
import cv2

model_name = 'faceNet'

def vgg_face(weights_path='vgg_face_weights.h5'):
    
    model = keras.Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    if weights_path:
        model.load_weights(weights_path)
    
    return model

def vgg_16():
    model = keras.applications.vgg16.VGG16()
    return model

def faceNet():
    model = load_model('facenet_keras.h5')
    return model

def get_model(name = 'faceNet'):
    if name == 'vgg_16': return vgg_16()
    elif name == 'vgg_face': return vgg_face()
    elif name == 'faceNet': return faceNet()
    else: return None

model = get_model(model_name)
face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_names = [layer.name for layer in model.layers]
epsilon_cos = 0.60 #cosine similarity
epsilon_dist = 120 #euclidean distance

def preprocess_image(img, model_name = 'faceNet'):
    if model_name == 'faceNet': 
        img = img.resize((160, 160))
        img = img_to_array(img)
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
    else: 
        img = img.resize((224, 224))
        img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return (a / (np.sqrt(b) * np.sqrt(c)))

def findCosineDifference(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
 
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def findL1Norm(source_representation, test_representation):
    l1_distance = source_representation - test_representation
    l1_distance = np.absolute(l1_distance)
    l1_distance = np.sum(l1_distance)
    return l1_distance

def verifyScore(euclidean_distance, cosine_similarity):
    if((euclidean_distance < epsilon_dist) & (cosine_similarity > epsilon_cos)): return True
    else: return False

def getFeatureVector(img, matrix = False, preprocess = True):
    if matrix: 
        img_representation = face_descriptor.predict(img)  
    else:
        if preprocess: img_representation = face_descriptor.predict(preprocess_image(img))[0,:]
        else: img_representation = face_descriptor.predict(img)[0,:]   
    return img_representation

def verifyFace(img1, img2, print_score = False):
    img1_representation = getFeatureVector(img1)
    img2_representation = getFeatureVector(img2)
    
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    l1_distance = findL1Norm(img1_representation, img2_representation)

    if print_score: print('Cos: ' + str(cosine_similarity) + ' Dist: ' + str(euclidean_distance) + ' L1 Dist: ' + str(l1_distance))
    cumulative_diff = cosineDifference*(euclidean_distance)*(l1_distance)

    print('cumulative_diff: ', cumulative_diff)

    return verifyScore(euclidean_distance, cosine_similarity)

def verifyFaceVector(img1, img2_representation, print_score = False):
    img1_representation = getFeatureVector(img)
    
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    l1_distance = findL1Norm(img1_representation, img2_representation)

    if print_score: print('Cos: ' + str(cosine_similarity) + ' Dist: ' + str(euclidean_distance) + ' L1 Dist: ' + str(l1_distance))

    return verifyScore(euclidean_distance, cosine_similarity)

def verifyVecs(img1_representation, img2_representation, print_score = False):
    
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    l1_distance = findL1Norm(img1_representation, img2_representation)

    if print_score: print('Cos: ' + str(cosine_similarity) + ' Dist: ' + str(euclidean_distance) + ' L1 Dist: ' + str(l1_distance))

    return verifyScore(euclidean_distance, cosine_similarity)

def verifyVecMat(vector, matrix, print_score = False, first_match = False, best_match = True):
    
    a = np.matmul(matrix, np.transpose(vector))
    b = np.sum(vector*vector)
    c = np.sum(matrix*matrix, axis = -1)
    cosine_similarity =  a / (np.sqrt(b) * np.sqrt(c))

    euclidean_distance = np.asarray(matrix) - np.asarray(vector)
    euclidean_distance = np.sum(euclidean_distance*euclidean_distance, axis = -1)
    euclidean_distance = np.sqrt(euclidean_distance)

    if print_score: print('Cos: '+str(cosine_similarity)+' Dist: '+str(euclidean_distance))
    if best_match: 
        e_by_c = cosine_similarity/(euclidean_distance*euclidean_distance)
        max_id = np.argmax(e_by_c)
        if((euclidean_distance[max_id]<epsilon_dist)&(cosine_similarity[max_id]>epsilon_cos)): return max_id
        else: return None 
    euclidean_distance =  [eu_dist < epsilon_dist for eu_dist in euclidean_distance] 
    cosine_similarity =  [cos_sim > epsilon_cos for cos_sim in cosine_similarity]
    match_vec = [euclidean_distance[i] & cosine_similarity[i] for i in range(len(cosine_similarity))]
    if first_match: 
        try: return match_vec.index(True)
        except ValueError: return None
    return match_vec
'''
def getConvList(img, layer = 2):
    layer_output = Model(inputs=model.layers[0].input, outputs=model.layers[layer].output)
    img_representation = layer_output.predict(preprocess_image(img))[0,:]
    images = []
    for channel in range(img_representation.shape[-1]):
        images.append(img_representation[:,:,channel])
    return images

def plotConvImage(images, index = 0):
    img = images[index]
    img = Image.fromarray(img)
    img.show()

def getConvImage(name, layer = 2, index = 0):
    image = model_images.readImage(name, cv_2 = True)
    image = model_images.getFaces(image)[0]
    images = getConvList(image, layer = 8)
    img = images[index]
    img = Image.fromarray(img)
    return img

def triggerFilter(layer_name, filter_index, input_img, alpha, iterations): 
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([input_img], [loss, grads])
    input_img_data = np.random.random((1, 224, 224, 3))* 20 + 128.
    for i in range(iterations):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * alpha
        if loss_value <= 0: break 
    return input_img_data

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    x = np.reshape(x, (x.shape[-1], x.shape[0], x.shape[1]))
    return x

def visualizeFilter(layer_name, filter_index, input_img = model.input, alpha = 1, iterations = 20):
    img = triggerFilter(layer_name, filter_index, input_img, alpha, iterations)[0]
    img = deprocess_image(img)
    img = Image.fromarray(img)
    return img

'''