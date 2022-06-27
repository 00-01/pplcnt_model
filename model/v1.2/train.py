from datetime import datetime
from glob import glob
import os

from keras import Input, Model
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten
from matplotlib import pyplot as plt
from numpy import interp
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.ops.confusion_matrix import confusion_matrix

################################ DATA ################################

DEBUG = 1
SAVE = 0

BATCH = 256
EPOCH = 256
ES = 128

MIN, MAX = 0, 255
CLASS = [0, 1]

################################################################ SETUP

## FUNCIONS
def log(l):
    if DEBUG == 1: print(l)


def draw_CM(label, predicted):
    cm = confusion_matrix(label, predicted)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # true : false rate
    true = 0
    false = 0
    for i, j in enumerate(label):
        if j != predicted[i]:
            false += 1
        else: true += 1
    classification_report = metrics.classification_report(label, predicted)
    multilabel_to_binary_matrics = metrics.multilabel_confusion_matrix(label, predicted)

    return plt.show(), print('true rate: ', true), print('false rate: ', false), print(), print('='*10, 'classification_report: ', '\n', classification_report), print('='*10, 'multilabel_to_binary_matrics by class_num: ', '\n', '[[TN / FP] [FN / TP]]',
                                                                                                                                                                       '\n', multilabel_to_binary_matrics)


def draw_ROC_AUC(x, y, category_names):
    n_classes = len(category_names)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y[:, i], x[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y.ravel(), x.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=1)

    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
             color='navy', linestyle=':', linewidth=1)

    colors = (['purple', 'pink', 'red', 'green', 'yellow', 'cyan', 'magenta', 'blue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1, label=f'Class {i} ROC curve (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([-.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC & AUC')
    plt.legend(loc="lower right")

    return plt.show()

################################################################ LOAD DATA

base_dir = "out"
img_list = glob(f"{base_dir}/*.png")
df = pd.read_csv(f"{base_dir}/output(err_dropped).csv")
log(df.head)

col = list(df.columns)
log(col)

## PATH TO REAL_PATH
for i in range(len(df)):
    df.iloc[i,0] = f"{base_dir}/{df.iloc[i,0]}"

## GET H,W
sample_img = Image.open(df.iloc[16,0])
img_array = np.array(sample_img, int)
H, W = img_array.shape

## DATASET TO TENSOR
data = []
label = []
for index, row in df.iterrows():
    img = Image.open(row[col[0]])
    img = data.append(list(img.getdata()))
    lbl = label.append(row[col[1]])
    if index % 200 == 0:  log(index)

data = np.array(data)
data = data.reshape(data.shape[0], H, W, 1)

label = np.array(label)
label = label.reshape(label.shape[0], 1)

########################################################################

# def decode(data, label):
#     image = tf.io.read_file(data)
#     decoded_image = tf.io.decode_png(image)
#     return decoded_image, label
#
#
# valid_dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
# valid_dataset = (valid_dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
#                  .padded_batch(BATCH)
#                  .prefetch(buffer_size=tf.data.AUTOTUNE))
#
# # view
# a, b = next(iter(valid_dataset))
#
# train_examples = data['x_train']
# train_labels = data['y_train']
# test_examples = data['x_test']
# test_labels = data['y_test']
#
# train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

########################################################################


################################################################ PRE-PROCESS DATA

## Nomalize
log(data[100:110])
norm_data = data/MAX
log(norm_data[100:110])

## Split
split = int(len(label)*0.9)
train_data, test_data = norm_data[:split], norm_data[split:]
train_label, test_label = label[:split], label[split:]

#######################################################################



################################ MODEL ################################

################################################################ BUILD

input = Input(shape=(H, W, 1))

x = Conv2D(128, (3,3))(input)
x = BatchNormalization()(x)
x = Activation('selu')(x)
x = Dropout(.1)(x)

x = Conv2D(64, (3, 3))(x)
x = BatchNormalization()(x)
x = Activation('selu')(x)
x = Dropout(.1)(x)

x = Conv2D(32, (3, 3))(x)
x = BatchNormalization()(x)
x = Activation('selu')(x)
x = Dropout(.1)(x)

x = Flatten()(x)

x = Dense(256, activation="selu")(x)
x = Dense(128, activation="selu")(x)
x = Dense(64, activation="selu")(x)

output = Dense(24, activation="softmax")(x)

model = Model(input, output)

################################################################ TRAIN

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

## fit
log_path = "logs/"+datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = k.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
es = EarlyStopping(monitor="val_loss", patience=ES, mode="auto", verbose=2)

history = model.fit(train_data, train_label, validation_split=0.1, batch_size=BATCH, epochs=EPOCH, verbose=1, callbacks=[es])  # callbacks=[es, tensorboard_callback])
print(history)

## plot
pd.DataFrame(history.history).plot(figsize=(16, 10), grid=1, xlabel="epoch", ylabel="accuracy")
plt.show()

################################################################ EVALUATE

loss, acc = model.evaluate(test_data, test_label, verbose=1)

predict = model.predict(test_data)
predicted = np.argmax(predict, axis=1)

################################################################ ACCURACY

## CM
draw_CM(test_label, predicted)

## ROC, AUC
x = label_binarize(predicted, classes=CLASS)
y = label_binarize(test_label, classes=CLASS)
draw_ROC_AUC(x, y, CLASS)

################################################################ SAVE

if SAVE == 1:
    file_name =  "model/light_detector_" + dt.now().strftime("%Y%m%d-%H%M%S")
    model_format = ".h5"
    model_name = file_name + model_format
    model.save(model_name)

########################################################################
