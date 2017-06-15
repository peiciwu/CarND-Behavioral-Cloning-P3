import csv
import cv2
import numpy as np
import sklearn
from random import shuffle
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU

def readDrivingLog(samples, dir_name, delimiter):
    with open(dir_name+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            for i in range(0, 3):
                filename = line[i].split(delimiter)[-1]
                line[i] = dir_name + '/IMG/' + filename # Add dir_name to the file name
            steering = float(line[3])
            samples.append((line[0], steering)) # center image
            samples.append((line[1], steering+0.25)) # left image
            samples.append((line[2], steering-0.25)) # right image

def generator(samples, batch_size):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            # Read images and steering angles for this batch
            images = []
            steerings = []
            for sample in batch_samples:
                filepath, steering = sample
                images.append(cv2.imread(filepath))
                steerings.append(steering)
                """
                # Flip image
                image_flipped = cv2.flip(image, 1)
                images.append(image_flipped)
                steerings.append(-steering)
                """

            X_train = np.array(images)
            y_train = np.array(steerings)
            yield sklearn.utils.shuffle(X_train, y_train)

def fit_model(train_generator, validation_generator, num_train_samples, num_validation_samples, EPOCH):
    """
    Use Nvidia model
    """
    model = Sequential()
    # Normalization
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    # Image cropping layer. Crop 70 pixels from the top and 25 pixels from the bottom.
    model.add(Cropping2D(cropping=((70,25), (0,0))))

    # Add three convolutional layers with a 5x5 kerenl and a 2x2 stride. Output depth is 24, 36, and 48, respectively.
    #model.add(Convolutional2D(24,5,5, subsample=(2,2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode='valid'))
    model.add(Activation('relu'))
    #model.add(ELU())
    model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode='valid'))
    model.add(Activation('relu'))
    #model.add(ELU())
    model.add(Convolution2D(48,5,5, subsample=(2,2), border_mode='valid'))
    model.add(Activation('relu'))
    #model.add(ELU())

    # Add two convolutional layers with a 3x3 kerenl. Output depth are both 64.
    model.add(Convolution2D(64,3,3, border_mode='valid'))
    model.add(Activation('relu'))
    #model.add(ELU())
    model.add(Convolution2D(64,3,3, border_mode='valid'))
    model.add(Activation('relu'))
    #model.add(ELU())

    model.add(Flatten())

    """
    FIXME. PW TEST
    # Fully connected layer. Output = 500
    """
    # Fully connected layer. Output = 100
    model.add(Dense(100))
    model.add(Activation('relu'))
    #model.add(ELU())
    # Fully connected layer. Output = 50
    model.add(Dense(50))
    model.add(Activation('relu'))
    #model.add(ELU())
    # Fully connected layer. Output = 10
    model.add(Dense(10))
    #model.add(ELU())
    model.add(Activation('relu'))

    # Final output layer
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(lr=0.001)) # Mean sqaure error and adam optimizer.
    history_object = model.fit_generator(train_generator,
                                         samples_per_epoch=num_train_samples,
                                         validation_data=validation_generator,
                                         nb_val_samples=num_validation_samples,
                                         nb_epoch=EPOCH, verbose=1)
    model.save('model.h5')

    return history_object
				
    
###########################
# TODO.
# - Data Augmentation: flip etc. DONE flip.
# - Using mulitple cameras.
# - Collect my own data.
###########################

#################################################################
# Start Main function
#################################################################

samples = []
readDrivingLog(samples, dir_name='data', delimiter='/') # Read from data provided by the class
#readDrivingLog(samples, dir_name='new_data', delimiter='\\') # Read my own data

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

### Use generator
BATCH_SIZE = 64
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

### Model building
EPOCH = 10
#history_object = fit_model(train_generator, validation_generator, len(train_samples)*3, len(validation_samples)*3, EPOCH)
history_object = fit_model(train_generator, validation_generator, len(train_samples), len(validation_samples), EPOCH)

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


