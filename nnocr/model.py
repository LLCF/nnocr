from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D

import os

char_list = ['S', 'X', 'U', 'D', 'C', '8', '7', '9', 'V', '0', 'B', '3', 'N', 'Q', '6', 
	'4', 'L', 'J', '1', 'Z', 'O', 'F', 'A', 'R', 'T', '2', 'K', 'E', 'Y', 'P', 'W', 'M', 'G', '5', 'H', 'I']

def get_char_classifier():
	model = Sequential()

	model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, 24, 24)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(36))
	model.add(Activation('softmax'))

	model.summary()

	optimizer = SGD()

	model.load_weights(os.path.join(os.path.dirname(__file__),'../weights/character_classifier.hdf5'))

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model
