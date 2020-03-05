'''
- Author      : Nikita Malviya
- Start Date  : 13/Feb/19
- Last Edited : 10/Apr/19 '''

# Dependencies(Modules ) :  Tkinter -> to create interface
#                           CV2 -> for image processing
#                           PIL -> for image procesing
#                           Glob -> to read file names from directory
#                           uuid -> to create ubique file name

import PIL
from PIL import Image,ImageTk
import cv2
from tkinter import *
import tkinter as tk 
from predict_emotion import predict
from glob import glob
import uuid

def image_to_photoimage(image):
	image= PIL.Image.fromarray(image)
	image = ImageTk.PhotoImage(image=image)
	return image

def show_frame():
	global image_extraction
	_, frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	frame = cv2.resize(frame, (width, height))
	frame = cv2.flip(frame, 1)

	image_extraction = predict(frame)
	frame = image_extraction.highlighted_face_image
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	#     frame = cv2.resize(frame, (300,300))

	imgtk = image_to_photoimage(frame)
	lmain.imgtk = imgtk
	lmain.configure(image=imgtk)
	lmain.after(10, show_frame)

def emotion_detection():
    
	home.destroy()
	submit.destroy()
	index = 0

	l11 = Label(root,image = emotion_images[index], borderwidth = 1, width = 100, height = 100 )
	l11.place(x = width + 40 , y = 20)
	l12 = Button(root, text = emotion_name[index].title(), font = ("Courier", 30),bg='black', 
	             fg = "white",  command = lambda: save_image(0), width = 8)
	l12.place(x = width + 160 , y = 50)
	index += 1

	l21 = Label(root,image = emotion_images[index], borderwidth = 1, width = 100, height = 100 )
	l21.place(x = width + 40 , y = 130)
	l22 = Button(root, text = emotion_name[index].title(), font = ("Courier", 30),bg='black', 
	             fg = "white",  command = lambda: save_image(1), width = 8)
	l22.place(x = width + 160 , y = 160)
	index += 1

	l31 = Label(root,image = emotion_images[index], borderwidth = 1, width = 100, height = 100 )
	l31.place(x = width + 40 , y = 240)
	l32 = Button(root, text = emotion_name[index].title(), font = ("Courier", 30),bg='black', 
	             fg = "white", command = lambda: save_image(2), width = 8)
	l32.place(x = width + 160 , y = 270)
	index += 1

	l41= Label(root,image = emotion_images[index], borderwidth = 1, width = 100, height = 100 )
	l41.place(x = width + 40 , y = 350)
	l42 = Button(root, text = emotion_name[index].title(), font = ("Courier", 30),bg='black', 
	             fg = "white",  command = lambda: save_image(3), width = 8)
	l42.place(x = width + 160 , y = 380)
	index += 1

	l51 = Label(root,image = emotion_images[index], borderwidth = 1, width = 100, height = 100 )
	l51.place(x = width + 40 , y = 460)
	l52 = Button(root, text = emotion_name[index].title(), font = ("Courier", 30),bg='black', 
	             fg = "white",  command = lambda: save_image(4), width = 8)
	l52.place(x = width + 160 , y = 490)
	index += 1

	l61 = Label(root,image = emotion_images[index], borderwidth = 1, width = 100, height = 100 )
	l61.place(x = width + 40 , y = 570)
	l62 = Button(root, text = emotion_name[index].title(), font = ("Courier", 30),bg='black', 
	             fg = "white", command = lambda: save_image(5), width = 8)
	l62.place(x = width + 160 , y = 600)
	index += 1
	show_frame()


def save_image(index):
	nface = image_extraction.nfaces
	if nface == 1:
	    frame = image_extraction.highlighted_face_image
	    face = image_extraction.faces
	    for (x,y,w,h) in face:
	        face_image = frame[y:y+h,x:x+w]
	    face_image = cv2.resize(face_image, (48,48))
	    outfile = '%s/%s/%s.png' % ("dataset",emotion_name[index], str(uuid.uuid4()))
	    cv2.imwrite(outfile,face_image)

width, height = 900,660
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = Tk()
root.configure(background = "#26D3EF")
root.title("Facial Emotion Recognition")
root.geometry("1366x768")
root.bind('<Escape>', lambda e: root.quit())

image_extraction = None
_, frame = cap.read()
frame = cv2.flip(frame, 1)
cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
img = PIL.Image.fromarray(cv2image)
imgtk = ImageTk.PhotoImage(image=img)
lmain = Label(root, image = imgtk, width = width, height = height)
lmain.place(x = 20, y = 20)

file_names = glob("image/icon/*")
emotion_images = []
emotion_name = []
for index in range(6):
	image = cv2.imread(file_names[index])
	image = cv2.resize(image, (100,100))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
	image = image_to_photoimage(image)
	emotion_images.append(image)
	name = file_names[index].split("/")[-1].split(".")[0]
	emotion_name.append(name)
    
    
    
home_image = cv2.imread("image/home.png")


home_image = cv2.resize(home_image,(1300,700))
home_image = cv2.cvtColor(home_image, cv2.COLOR_BGR2RGBA)
home_image = image_to_photoimage(home_image)
home = Label(root, image = home_image, width = 1300, height =700)
home.place(x = 0, y = 0)

submit = Button(root,text = "Check it out!", command = emotion_detection, height = 2, font = ("Courier", 20), 
                bg = "black", fg = "white")



submit.place(x = 880,y= 440)
# emotion_detection()

root.mainloop()
cap.release()
