import cv2
import glob

faceDet = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_default.xml") #get xml files required
faceDet2= cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_alt2.xml")
faceDet3= cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_alt.xml")
faceDet4= cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_alt_tree.xml")
emotions = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"] #define emotions
def detect_faces(emotion):
	files = glob.glob("./arranged_images/%s/*" %emotion) #define file(s) location
	print (files)
	filenumber = 0
	for f in files: 
		frame = cv2.imread(f, cv2.IMREAD_GRAYSCALE) #read as black and white
	
		gray = frame
		face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
		face2= faceDet2.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=10, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
		face3= faceDet3.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=10, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
		face4= faceDet4.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=10, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)

		# go over detected faces, stop at first detected face, return empty if no face.
		if len(face) == 1:
			facefeatures = face
		elif len(face2) == 1:
			facefeatures = face2
		elif len(face3) == 1:
			facefeatures = face3
		elif len(face4) == 1:
			facefeatures = face4
		else:
			facefeatures = ""

		# crop face from image and save in defined path
		for (x,y,w,h) in facefeatures:
			print ("face found in file: %s" %f)
			gray = gray[y:y+h, x:x+w]

			try:
				out = cv2.resize(gray, (270,270))
				cv2.imwrite("./training_images/%s/%s.jpg" %(emotion, filenumber), out) #define path and write
			except:
				pass # if error, pass file
			filenumber += 1 # increment image number

for emotion in emotions:
			detect_faces(emotion)