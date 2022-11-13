# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 19:50:51 2021

@author: sas11
"""

from tensorflow import keras
import tensorflow as tf # Imports tensorflow
import tensorflow_addons as tfa


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D,LSTM
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tensorflow.keras.metrics import Recall,Precision,AUC,TruePositives,TrueNegatives,FalseNegatives,FalsePositives
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import cv2
from glob import glob


img_shape = (48, 48, 3)
data_dir="df"

Name = "cnn1"
in_image = []
lebel = []
rel_dirname = os.path.dirname(__file__)
    
for dirname in os.listdir(os.path.join(rel_dirname, data_dir)):
        for filename in glob(os.path.join(rel_dirname, data_dir+'/'+dirname+'/*.jpg')):
             img = image.load_img(os.path.join(rel_dirname, filename),target_size=img_shape)
             img = image.img_to_array(img)
             img = img/255.0
             in_image.append(img)
             lebel.append(dirname)

X = np.array(in_image)
lebel = np.array(lebel)
y=to_categorical(lebel)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(48,48,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#

model.add(Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))
model.add(Conv2D(filters=128, kernel_size=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))




model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(2, activation='softmax'))
print(model.summary())

model.compile(optimizer= keras.optimizers.Adam(),  loss=keras.losses.BinaryCrossentropy() , metrics=['acc',Recall(),Precision(),AUC(),TruePositives(),TrueNegatives(),FalseNegatives(),FalsePositives()])
plot_model(model, to_file=Name+'.png',show_shapes= True , show_layer_names=True)
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=64)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower left')
plt.savefig(Name+'acc.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(Name+'loss.png')
plt.show()

plt.plot(history.history['recall_1'])
plt.plot(history.history['val_recall_1'])
plt.title('Model recall')
plt.ylabel('recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(Name+'recall.png')
plt.show()


plt.plot(history.history['precision_1'])
plt.plot(history.history['val_precision_1'])
plt.title('Model Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(Name+'precision.png')
plt.show()


model.save(Name+'.h5')

pd.DataFrame.from_dict(history.history).to_csv(Name+'.csv',index=False)