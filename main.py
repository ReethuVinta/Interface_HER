from flask import Flask
from flask import render_template,redirect, url_for, request
from flask_caching import Cache
from werkzeug.utils import secure_filename
import os
import numpy as np
from tqdm import trange
from keras.models import  load_model
import cv2
import keras
import glob
from keras import backend as K
import dlib
from imutils import face_utils
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage
from imutils import face_utils
global shape_x
global shape_y
global input_shape
global nClasses
import warnings 
warnings.simplefilter('ignore')
emotions = {0:"Angry",1:"Neutral",2:"Happy",3:"Sad"}
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

shape_x = 48
shape_y = 48
input_shape = (shape_x, shape_y, 1)
nClasses = 7
thresh = 0.25
frame_check = 20
##############################################################
def GetVideoFeatures(Path,basepath,Xmodel,face_detect,predictor_landmarks):
	cap = None
	cap = cv2.VideoCapture(Path)
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))//10
	frame_number=0
	image_number=1
	Video_Features=[]
	while cap.isOpened():
		ret, frame = cap.read()
		try :
			if(frame_number % length == 0 and image_number < 9 ):
				cv2.imwrite(os.path.join(basepath,"static",secure_filename("Img_{}.jpg".format(image_number))),frame)
				image_number+=1
				
			frame_number+=1
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

config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "simple", # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 10
}
app = Flask(__name__)
app.config.from_mapping(config)
cache = Cache(app)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config["UPLOAD_FOLDER"] = '/Upload'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/',methods=['GET'])
def index():
	print("bye")
	return render_template('base.html')

@app.route('/video',methods=["POST"])
def video():
	if request.method == "POST":
		predictions=[[1000,1000,1000,1000]]
		emotion = None
		print("iiiiii")
		if request.files:
			print("iiiiii")
			f = request.files['file']
			basepath = os.path.dirname(__file__)
			file_path = os.path.join(
				basepath, 'Upload', secure_filename(f.filename))
			f.save(file_path)

			print("1")
			Xmodel = load_model('./models/video.h5')
			print("2")
			face_detect = dlib.get_frontal_face_detector()
			print("3")
			predictor_landmarks  = dlib.shape_predictor("./models/face_landmarks.dat")
			print("4")
			model=load_model('./models/dummy.hdf5')
			print("5")
			
			video_features=GetVideoFeatures(file_path,basepath,Xmodel,face_detect,predictor_landmarks)
			video_features=np.expand_dims(video_features,axis=0)
			predictions=model.predict(video_features)
			predictions=predictions.tolist()
			predictions= predictions[0] 
			maximum = predictions.index(max(predictions))
			emotion = emotions[maximum]
			predictions=[100*x for x in predictions]
			predictions_round = [ '%.2f' % elem for elem in predictions ]
			# predictions=[1.00,1.00,1.00,1.00]
			print(predictions_round)
		return render_template('base.html',prob=predictions_round,emotion = emotion)
	return "OK"

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response