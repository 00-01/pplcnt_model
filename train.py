import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from helper import path

INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32

# print("[INFO] loading dataset...")
rows = open(path.ANNOTS_PATH).read().strip().split("\n")
data = []
targets = []
filenames = []
for row in rows:
    # row = row.split(",")
    # (filename, startX, startY, endX, endY) = row
    (filename, startX, startY, endX, endY) = row.split(",")

    # derive the path to the in image, load the image (in OpenCV format), and grab its dimensions
    imagePath = os.path.join(path.IMAGES_PATH, filename)
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    # scale bounding box coordinates relative to spatial dimensions of in image
    startX = float(startX) / w
    startY = float(startY) / h
    endX = float(endX) / w
    endY = float(endY) / h
    # load the image and preprocess it
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	# update our list of data, targets, and filenames
	data.append(image)
	targets.append((startX, startY, endX, endY))
	filenames.append(filename)

# normalize
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

# in, test split
split = train_test_split(data, targets, filenames, test_size=0.10, random_state=42)
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]
# save test filename
with open(path.TEST_FILENAMES, "w") as f:
    f.write("\n".join(testFilenames))

# load model
vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
vgg.trainable = False  # not trainable
flatten = vgg.output  # flatten the max-pooling out of VGG
flatten = Flatten()(flatten)
# make dense layer header to out predicted bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)

model = Model(inputs=vgg.input, outputs=bboxHead)

model.compile(loss="mse", optimizer=Adam(lr=INIT_LR))
model.summary()

H = model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets),
              batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1)

model.save(path.MODEL_PATH, save_format="h5")

# plot
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(path.PLOT_PATH)

