'''
- Author      : Nikita Malviya
- Start Date  : 13/Feb/19
- Last Edited : 10/Apr/19 '''

import cv2
import numpy as np

def preprocess_image(image, scaling = True):
    image = image.astype('float32')
    image = image / 255.0
    if scaling:
        image = image - 0.5
        image = image * 2.0
    return image

def preprocess_live_image(face_image, normalize = True):
	face_image = cv2.resize(face_image, (48, 48))
	face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
	face_image = np.expand_dims(face_image, -1)
	if normalize:
		face_image = preprocess_image(face_image)
	return face_image
