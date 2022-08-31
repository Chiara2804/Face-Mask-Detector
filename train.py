from keras.optimizers import RMSprop #pip install keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split #pip install -U scikit-learn
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils #pip install imutils
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2 #pip install opencv-python

model = Sequential([
    Conv2D(100, (3, 3), activation = 'relu', input_shape = (150, 150, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(100, (3, 3), activation = 'relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dropout(0.5),
    Dense(50, activation = 'relu'),
    Dense(2, activation = 'softmax')
])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

TRAINING_DIR = "Dataset/train"
train_datagen = ImageDataGenerator(rescale = 1.0 / 255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest')

train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                    batch_size = 100, 
                                                    target_size = (150, 150))
VALIDATION_DIR = "Dataset/test"
validation_datagen = ImageDataGenerator(rescale = 1.0 / 255)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                         batch_size = 30, 
                                                         target_size = (150, 150))
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose = 0, save_best_only = True, mode = 'auto')

#pip install git+git://github.com/fchollet/keras.git --upgrade --no-deps
history = model.fit_generator(train_generator,
                              epochs = 30,
                              validation_data = validation_generator,
                              callbacks = [checkpoint])


