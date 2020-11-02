from flask import Flask
from flask import render_template,redirect, url_for, request
from werkzeug.utils import secure_filename
import os
# from modelcode import *

from numpy import save
import torch
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


######################################################################

#####################################################################
def GetVideoFeatures(Path,Xmodel):
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
				print("preeeeee")
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
	if request.method == "POST":
		print("hi")
		if request.files:
			f = request.files['file']
			basepath = os.path.dirname(__file__)
			file_path = os.path.join(
				basepath, 'Upload', secure_filename(f.filename))
			f.save(file_path)
			Xmodel = load_model('./models/video.h5')
			face_detect = dlib.get_frontal_face_detector()
			predictor_landmarks  = dlib.shape_predictor("./models/face_landmarks.dat")
            model = load_model('./models/dummy.hdf5')
			predictions=None
			video_features=GetVideoFeatures(video_path,Xmodel)
			predictions=model.predict(video_features)
			print(predictions.shape)
			_, predicted = torch.max(predictions.data, 1)
			print(predicted)
	return render_template('index.html',prob=predictions)
