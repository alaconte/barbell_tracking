import ctypes
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\bin\\cudart64_110.dll")
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\bin")
os.add_dll_directory("C:\\Program Files\\NVIDIA\\CUDNN\\v8.3\\bin")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

import random
import pickle

batch_size = 32
epochs = 25


def generate_data():
    # Load data
    with open('labeled_data.pkl', 'rb') as f:
        labeled_data = pickle.load(f)

    random.shuffle(labeled_data)

    # partition training set into training and validation set
    train = labeled_data[0:1000]
    test = labeled_data[1001:]

    return train, test

def base_cnn_activation(activation):
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
  model.add(Activation(activation))
  model.add(Conv2D(32, (3, 3), padding='same'))
  model.add(Activation(activation))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(Activation(activation))
  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(Activation(activation))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation(activation))
  model.add(Dropout(0.5))
  model.add(Dense(units=2))
  model.add(Activation('softmax'))
  opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
  model.compile(loss='mean_absolute_error',
                optimizer=opt,
                metrics=['accuracy'])
  print(model.summary())

  return model

train, test = generate_data()

x_train = np.array([item[0] for item in train])
y_train = np.array([item[1:] for item in train])

x_validate = np.array([item[0] for item in test])
y_validate = np.array([item[1:] for item in test])

save_best_model = ModelCheckpoint('best_model.{}'.format('softmax'), monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
model =  base_cnn_activation('softmax')

history_activations = dict()
history_activations['sigmoid'] = model.fit(x_train, y_train,
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              validation_data=(x_validate, y_validate),
                                              shuffle=True,
                                              callbacks=[save_best_model])

model.save("best_model.h5")

# Plot training accuracy
plt.plot(history_activations['sigmoid'].history['accuracy'], 'o-', label='CNN')

plt.title('training accuracy')
plt.ylabel('training accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
plt.show()

# Plot validation accuracy
plt.plot(history_activations['sigmoid'].history['val_accuracy'], 'o-', label='CNN')
plt.title('validation accuracy')
plt.ylabel('validation accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
plt.show()