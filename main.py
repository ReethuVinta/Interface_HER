from flask import Flask
from flask import render_template,redirect, url_for, request
from werkzeug.utils import secure_filename
import os
# from modelcode import *

from numpy import save

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from keras.models import Sequential , load_model
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, LocallyConnected2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2
from keras.utils import plot_model, to_categorical
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input
from keras.layers import concatenate
from pathlib import Path
from torch.autograd import Variable
import pandas as pd
import librosa
import cv2
import os
import keras
from scipy.io import wavfile
import soundfile as sf
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import backend as K
import dlib
from imutils import face_utils
### Image processing ###
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage
import dlib
from tensorflow.keras.models import load_model
from imutils import face_utils
import requests
from tensorflow.python.client import device_lib
import torch.nn as nn
import torch.nn.functional as F
import torch
global shape_x
global shape_y
global input_shape
global nClasses
global predictions
import warnings 
warnings.simplefilter('ignore')

def pad(frames):
	frames = [x.tolist() for x in frames]
	if(len(frames)>100):
		frames = frames[len(frames)-100:]
	else:
		temp = 100 - len(frames)
	for _ in range(temp):
		frames.append([0]*1024)
	frames=np.array(frames)
	return frames

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
(eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
shape_x = 48
shape_y = 48
input_shape = (shape_x, shape_y, 1)
nClasses = 7
thresh = 0.25
frame_check = 20
##############################################################
def GetVideoFeatures(Path,Xmodel,face_detect,predictor_landmarks):
	cap = None
	cap = cv2.VideoCapture(Path)
	Video_Features=[]
	while cap.isOpened():
		ret, frame = cap.read()
		try :
			face_index = 0    
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			rects = face_detect(gray, 1)
				#gray, detected_faces, coord = detect_face(frame)

			for (i, rect) in enumerate(rects):
				# print("hhhhh")
				shape = predictor_landmarks(gray, rect)
				shape = face_utils.shape_to_np(shape)
					
					# Identify face coordinates
				(x, y, w, h) = face_utils.rect_to_bb(rect)
				face = gray[y:y+h,x:x+w]
					
					#Zoom on extracted face
				face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))
					
					#Cast type float
				face = face.astype(np.float32)
					
				
				face /= float(face.max())
				face = np.reshape(face.flatten(), (1, 48, 48, 1))
				# print(face.shape)      
				get_3rd_layer_output = K.function([Xmodel.layers[0].input],
									[Xmodel.layers[-2].output])
				layer_output = get_3rd_layer_output([face])[0]
				Video_Features.append(layer_output[0])
		except:
			Video_Features.append(np.zeros(1024,dtype=np.float32))
			break
		
	cap.release()
	cv2.destroyAllWindows()
	return pad(Video_Features)
#####################################################
# def video_analysis(video_path,Xmodel,model):

	
# #####################################################
# print(video_analysis('./neutraltest1.mp4'))
# print(video_analysis('./angry1.mp4'))
# print(video_analysis('./sad1.mp4'))
# print(video_analysis('./sad2.mp4'))
# print(video_analysis('./happy1.mp4'))
# print([[1]*10]*2)
 

app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config["UPLOAD_FOLDER"] = '/Upload'

@app.route('/',methods=['GET'])
def index():
	print("bye")
	return render_template('base.html')

@app.route('/video',methods=['GET','POST'])
def video():
	predictions=[[1000,1000,1000,1000]]
	if request.method == "POST":
		print("iiiiii")
		if request.files:
			print("iiiiii")
			predictions=[[1000,1000,1000,1000]]
			f = request.files['file']
			# 	basepath = os.path.dirname(__file__)
			# 	file_path = os.path.join(
			# 		basepath, 'Upload', secure_filename(f.filename))
			# 	f.save(file_path)
			# print("1")
			# Xmodel = load_model('./models/video.h5')
			# print("2")
			# face_detect = dlib.get_frontal_face_detector()
			# print("3")
			# predictor_landmarks  = dlib.shape_predictor("./models/face_landmarks.dat")
			# print("4")
			# model=load_model('./models/dummy.hdf5')
			# print("5")
			
			# video_features=GetVideoFeatures(file_path,Xmodel,face_detect,predictor_landmarks)
			# video_features=np.expand_dims(video_features,axis=0)
			# predictions=model.predict(video_features)
			# predictions=predictions.tolist()
			# predictions=[str(x)for x in predictions] 
			predictions=[[1,2,3,4]]
			print("aaaa")
			print(predictions)
			print("here")
	print(predictions)
	return render_template('index.html',prob=predictions)

# @app.route('/video_a',methods=['GET'])
# def video_a():
# 	return render_template('index.html',prob=predictions)