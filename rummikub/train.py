from os import listdir
from os.path import expanduser
from random import shuffle

from matplotlib import pyplot
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix


# Get training directory contents
training_image_files = listdir(expanduser('~/Desktop/rummikub/images/training/'))
shuffle(training_image_files)

training_image_files2 = [i.split('_') for i in training_image_files]


# Create list of classes
classes = []
for index, _ in enumerate(training_image_files2):
	classes.append(training_image_files2[index][0] + '_' + training_image_files2[index][1])


encoder = LabelEncoder()
encoder.fit(classes)
encoded_Y = encoder.transform(classes)
dummy_y = np_utils.to_categorical(encoded_Y)



# Create 4d array of image data
training_image_data = []

for image_name in training_image_files:
	training_image_data.append(img_to_array(load_img(expanduser(f'~/Desktop/rummikub/images/training/{image_name}'))))

stacked_training_images = np.asarray(training_image_data)




# define cnn model
def baseline_model():
	model = Sequential()
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(170, 120, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(BatchNormalization())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dense(75, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dense(54, activation='sigmoid'))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model


model = baseline_model()
model.summary()
model.fit(stacked_training_images, dummy_y, epochs=7, batch_size=5, shuffle = True)
model.save("big_model")



##### See how model does in test set

testing_image_files = listdir(expanduser('~/Desktop/rummikub/images/testing/'))
shuffle(testing_image_files)
testing_image_files2 = [i.split('_') for i in testing_image_files]


# Create list of test_classes
test_classes = []
for index, _ in enumerate(testing_image_files2):
	test_classes.append(testing_image_files2[index][0] + '_' + testing_image_files2[index][1])



# Create 4d array of image data
test_image_data = []

for image_name in testing_image_files:
	test_image_data.append(img_to_array(load_img(expanduser(f'~/Desktop/rummikub/images/testing/{image_name}'))))

stacked_test_images = np.asarray(test_image_data)



test_set_results = model.predict_classes(stacked_test_images)
real_test_classes = encoder.transform(test_classes)


accuracy_score(real_test_classes, test_set_results)

cm = confusion_matrix(real_test_classes, test_set_results)
cm = cm/200


pd.DataFrame({'labels': encoder.classes_, 'accuracy': cm.diagonal()})
