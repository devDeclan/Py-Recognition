# set the matplotlib backend so figures can be saved in the background
import argparse
import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import lib.config as config



train_data = pd.read_csv(config.TRAIN_DATA)
print(train_data.head())
test_data = pd.read_csv(config.TEST_DATA)
print(test_data.head())

X_train = []
y_train = []
X_test = []
y_test = []

# for loop to read and store train frames
for i in tqdm(range(train_data.shape[0])):
  # loading the images and keeping the target size as (224, 224, 3)
  img = image.load_img(train_data["image"][i], target_size=(224, 224, 3))
  # converting it to array
  img = image.img_to_array(img)
  # normalizing the pixel value
  img = img/255
  # storing each image in array X
  X_train.append(img)
X_train = np.array(X_train)
y_train = train_data["label"]
y_train = np.array(y_train)

# creating dummies of target variable for train set
y_train = pd.get_dummies(y_train)

# for loop to read and store test frames
for i in tqdm(range(test_data.shape[0])):
  # loading the images and keeping the target size as (224, 224, 3)
  img = image.load_img(test_data["image"][i], target_size=(224, 224, 3))
  # converting it to array
  img = image.img_to_array(img)
  # normalizing the pixel value
  img = img/255
  # storing each image in array X
  X_test.append(img)
X_test = np.array(X_test)
y_test = test_data["label"]
y_test = np.array(y_test)

# creating dummies of target variable for test set
y_test = pd.get_dummies(y_test)


# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print(y_train)
print(y_test)


## Defining the architecture of the video classification model

# creating the base model of pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# extracting features for training frames
X_train = base_model.predict(X_train);
print(X_train.shape)

# extracting features for training frames
X_test = base_model.predict(X_test);
print(X_test.shape)

# print(X_train.shape)
train_rows = X_train.shape[0]
print(train_rows)
test_rows = X_test.shape[0]
print(test_rows)

# reshaping the training frames in single dimension
X_train = X_train.reshape(train_rows, 7 * 7 * 512)
print(X_train.shape)
X_test = X_test.reshape(test_rows, 7 * 7 * 512)
print(X_test.shape)

# normalizing the pixel values
max = X_train.max()
X_train = X_train/max
print(X_train.shape)
X_test = X_test/max
print(X_test.shape)


model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(7 * 7 * 512, )))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
print(model.summary())

save_check_point = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[save_check_point], batch_size=20)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('lit.hdf5')