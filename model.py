import csv
import cv2
import numpy as np
import sklearn
from random import shuffle

def readDrivingLog(dir_name, samples):
    with open(dir_name+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip the header
        for line in reader:
            line[0] = dir_name + '/' + line[0] # Add dir_name to the file name
            samples.append(line)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            # Read images and steering angles for this batch
            images = []
            steerings = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0])
                images.append(image)
                steerings.append(float(batch_sample[3]))

            X_train = np.array(images)
            y_train = np.array(steerings)
            yield sklearn.utils.shuffle(X_train, y_train)
				
    
###########################
# TODO.
# - Data Augmentation: flip etc.
# - Using mulitple cameras.
# - Collect my own data.
###########################

samples = []
readDrivingLog('data', samples) # Read from samples data provided by the class

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

### Use generator
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

### Model building
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D
#from keras.layers.advanced_activations import ELU
from keras.layers import Cropping2D

"""
Nvidia model
"""
model = Sequential()
### Normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
### Image cropping layer. Crop 70 pixels from the top and 25 pixels from the bottom.
model.add(Cropping2D(cropping=((70,25), (0,0))))

### Add three convolutional layers with a 5x5 kerenl and a 2x2 stride. Output depth is 24, 36, and 48, respectively.
#model.add(Convolutional2D(24,5,5, subsample=(2,2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode='valid', activation='relu'))
#model.add(ELU())
model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode='valid', activation='relu'))
#model.add(ELU())
model.add(Convolution2D(48,5,5, subsample=(2,2), border_mode='valid', activation='relu'))
#model.add(ELU())

### Add two convolutional layers with a 3x3 kerenl. Output depth are both 64.
model.add(Convolution2D(64,3,3, border_mode='valid', activation='relu'))
#model.add(ELU())
model.add(Convolution2D(64,3,3, border_mode='valid', activation='relu'))
#model.add(ELU())

model.add(Flatten())

### Fully connected layer. Output = 100
model.add(Dense(100))
#model.add(Activation('relu'))
### Fully connected layer. Output = 50
model.add(Dense(50))
#model.add(Activation('relu'))
### Fully connected layer. Output = 10
model.add(Dense(10))
#model.add(Activation('relu'))

### Final output layer
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam') # Mean sqaure error and adam optimizer.
history_object = model.fit_generator(train_generator,
        samples_per_epoch=len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

### Plot the training and validation loss for each epoch
import matplotlib
matplotlib.use('agg') # For remote ssh
import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('history.png')

model.save('model.h5')

