# Author                 :  Mohammad Noor
# Start Date             :  13/Feb/19

# Dependencies(Modules ) :  cv2

# Last Edited            :  15/Apr/19

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')

def find_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find faces in image
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(100,100),flags=cv2.CASCADE_SCALE_IMAGE )

    # print number of faces detected in the image
    # get bounding box for each detected face
    for (x,y,w,h) in faces:
        # add bounding box to color image
        x -= 2
        y -= 2
        w += 4
        h += 4
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # convert BGR image to RGB for plotting
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # return the image, along with bounding box
    return img, faces