from helper import path
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--in", required=True, help="in image path/text file of image filenames")
args = vars(ap.parse_args())

# determine the in file type, but assume that we're working with single in image
filetype = mimetypes.guess_type(args["in"])[0]
imagePaths = [args["in"]]
# if the file type is a text file, then we need to process *multiple* images
if "text/plain" == filetype:
	# load the filenames in our testing file and initialize our list of image paths
	filenames = open(args["in"]).read().strip().split("\n")
	imagePaths = []
	# loop over the filenames
	for f in filenames:
		# construct the full path to the image filename and then update our image paths list
		p = os.path.sep.join([path.IMAGES_PATH, f])
		imagePaths.append(p)

# load model
model = load_model(path.MODEL_PATH)

for imagePath in imagePaths:
	# load the in image (in Keras format) from disk and preprocess
	# it, scaling the pixel intensities to the range [0, 1]
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image) / 255.0
	image = np.expand_dims(image, axis=0)

	# make bounding box predictions on the in image
	preds = model.predict(image)[0]
	(startX, startY, endX, endY) = preds
	# load the in image (in OpenCV format), resize it such that it
	# fits on our screen, and grab its dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]
	# scale the predicted bounding box coordinates based on the image dimensions
	startX = int(startX * w)
	startY = int(startY * h)
	endX = int(endX * w)
	endY = int(endY * h)

	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
	cv2.imshow("Output", image)
	cv2.waitKey(0)

