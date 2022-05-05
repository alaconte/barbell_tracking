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
  """
  TO-DO: Copy the code from the base_cnn() function above and paste it here.
  This starter code sets the activation function to 'relu' by default. Modify
  the code so that it can work with an user-supplied activation functions instead
  of the default 'relu' activation. Do not change the 'softmax' activation. Refer
  to the simulation code below to understand the possible values that the input
  'activation' may take.
  """
  """
  Define a convolutional neural network using the Sequential model. This is the 
  basic CNN that you will need to reuse for the remaining parts of the assignment.
  It would be good to familiarize yourself with the workings of this basic CNN.
  """
  model = Sequential()
  '''
  Add 2D convolution layers the perform spatial convolution over images. This 
  layer creates a convolution kernel that is convolved with the layer input to 
  produce a tensor of outputs. When using this layer as the first layer in a 
  model, provide the keyword argument 'input_shape' (tuple of integers). Besides,
  the Conv2D function takes as input
  - filters: Integer, the dimensionality of the output space (i.e. the number of
   output filters in the convolution). We set it to 32.
  - kernel_size: An integer or tuple/list of 2 integers, specifying the height
   and width of the 2D convolution window. Can be a single integer to specify 
   the same value for all spatial dimensions. We set it to (3, 3).

  Here, we create a stack of (CONV2D, Activation, CONV2D, Activation) layers with 
  the ReLu activation function 
  '''
  model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
  model.add(Activation(activation))
  model.add(Conv2D(32, (3, 3), padding='same'))
  model.add(Activation(activation))
  '''
  Perform MaxPooling operation for 2D spatial data. This downsamples the input
  along its spatial dimensions (height and width) by taking the maximum value 
  over an input window of size 2X2 for each channel of the input.
  '''
  model.add(MaxPooling2D(pool_size=(2, 2)))
  '''
  Add a Dropout layer that  randomly sets input units to 0 with a frequency of
  'rate' at each step during training time, which helps prevent overfitting. 
  Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all
  inputs is unchanged. We set the rate to 0.25 for Dropout.
  '''
  model.add(Dropout(0.25))
  '''
  Create another stack of (CONV2D, Activation, CONV2D, Activation) layers with 
  the ReLu activation function. Set the 'filters' to 64.
  '''
  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(Activation(activation))
  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(Activation(activation))
  '''
  Perfrom MaxPooling and Dropout similar to the one defined earlier.
  '''
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  '''
  The image is still in 3D. It needs be unrolled from 3D to 1D using the Flatten
  layer. Then add a Dense layers on top of it followed by ReLu activation and 
  dropout of 0.5. This helps to create a fully-connected layer.
  '''
  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation(activation))
  model.add(Dropout(0.5))
  '''
  Create the output layer using the Dense layer with 'softmax' activation. The 
  number of predicted output needs to be equal to 'num_classes'.
  '''
  model.add(Dense(units=2))
  model.add(Activation('softmax'))
  '''
  Set the optimizer for doing mini-batch gradient descent. Here, we make use of 
  the RMSprop optimizer that comes with Keras. We supply some default values for
  the parameters learning_rate and decay. Do not modify them.
  '''
  opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
  '''
  Compile the model for training. Since this is a multi-class classification 
  problem, we use the 'categorical_crossentropy' loss function and 'accuracy' as
  the desired performance metric.
  '''
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