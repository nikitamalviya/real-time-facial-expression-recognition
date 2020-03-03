# Author                 :  Nikita Malviya
# Start Date             :  13/Feb/19

# Dependencies(Modules ) :  cv2

# Last Edited            :  10/Apr/19

import cv2
from preprocess import preprocess_live_image
from find_face import find_face 
from model import model

emotion_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def predict(original_image):
	model.load_weights("models/_mini_XCEPTION.87-0.65.hdf5")
	highlighted_face_image, faces = find_face(original_image)
	nfaces = len(faces)
	prediction = None
	predicted_emotion = None
	if len(faces) > 0:
		for [x, y, w, h] in faces:
			image = original_image[y:y+h,x:x+w]
			image = preprocess_live_image(image)
			image = [[image]]
			prediction = model.predict(image)
			maximum = max(prediction)
			predicted_emotion = emotion_list[prediction.argmax()]
			cv2.putText(highlighted_face_image, predicted_emotion, (x,y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 10)
	return image_extraction(highlighted_face_image, faces, nfaces, prediction, predicted_emotion,original_image)

class image_extraction:
	def __init__(self, highlighted_face_image, faces, nfaces, prediction, predicted_emotion,original_image):
		self.highlighted_face_image = highlighted_face_image
		self.faces = faces
		self.nfaces = nfaces
		self.prediction = prediction
		self.predicted_emotion = predicted_emotion 
		self.original_image = original_image