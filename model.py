import csv
import cv2
import numpy as np

# Read driving data cvs file.
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Read images and steer measurements.
images = []
measurements = []
for line in lines[1:-1]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

###########################
# TODO.
# - Data Augmentation: flip etc.
# - Using mulitple cameras.
# - Collect my own data.
###########################

X_train = np.array(images)
y_train = np.array(measurements)

# Model building
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D
#from keras.layers.advanced_activations import ELU
from keras.layers import Cropping2D


"""
Simple model

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) # mean center normalization
model.add(Flatten())
model.add(Dense(1))
"""

"""
Nvidia model
"""
model = Sequential()
# Normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# Image cropping layer. Crop 70 pixels from the top and 25 pixels from the bottom.
model.add(Cropping2D(cropping=((70,25), (0,0))))

# Add three convolutional layers with a 5x5 kerenl and a 2x2 stride. Output depth is 24, 36, and 48, respectively.
#model.add(Convolutional2D(24,5,5, subsample=(2,2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode='valid', activation='relu'))
#model.add(ELU())
model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode='valid', activation='relu'))
#model.add(ELU())
model.add(Convolution2D(48,5,5, subsample=(2,2), border_mode='valid', activation='relu'))
#model.add(ELU())

# Add two convolutional layers with a 3x3 kerenl. Output depth are both 64.
model.add(Convolution2D(64,3,3, border_mode='valid', activation='relu'))
#model.add(ELU())
model.add(Convolution2D(64,3,3, border_mode='valid', activation='relu'))
#model.add(ELU())

model.add(Flatten())

# Fully connected layer. Output = 100
model.add(Dense(100))
#model.add(Activation('relu'))
# Fully connected layer. Output = 50
model.add(Dense(50))
#model.add(Activation('relu'))
# Fully connected layer. Output = 10
model.add(Dense(10))
#model.add(Activation('relu'))

# Final output layer
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam') # Mean sqaure error and adam optimizer.
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, verbose=2) # Split 20% as validation set.
#### TODO. Visualizing Loss ####

model.save('model.h5')

