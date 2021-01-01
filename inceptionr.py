# -*- coding: utf-8 -*-
"""InceptionR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EcjQ92jJyZdbiQHcSKLLROdoU660Vt0E
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import Model
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
  
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False
  
# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
# print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.3
x = layers.Dropout(0.4)(x)        
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.3
x = layers.Dropout(0.4)(x)            
x = layers.Dense(512, activation='relu')(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0005), 
              loss = 'binary_crossentropy', 
              metrics = ['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import os
import zipfile

from google.colab import drive
drive.mount('/content/drive')
#Checking the root folder
!pwd

train_data_dir = "Training"

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                   rescale=1/255, 
                                   validation_split=0.2, 
                                   rotation_range = 5,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


train_generator = train_datagen.flow_from_directory(train_data_dir,subset='training',target_size = (150, 150),
batch_size = 5,  class_mode = 'binary')
validation_generator = train_datagen.flow_from_directory(train_data_dir,subset='validation',target_size = (150, 150),
batch_size = 5, class_mode = 'binary')

history = model.fit_generator(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 30,
            epochs = 100,
            validation_steps = 10)

            #verbose = 2)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
# plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'y', label='Validation accuracy')
plt.ylim(0.3,1.005)
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

# plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.ylim(0,2.3)
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

plt.show()