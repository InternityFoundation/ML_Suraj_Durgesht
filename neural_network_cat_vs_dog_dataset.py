# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 00:16:25 2019

@author: Suraj
"""


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialising the CNN
classifier = Sequential()

# Step 1 Convulution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#polling
#classifier.add(MaxPooling2D(pool_size = (2, 2,)))

# adding second Convulution layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2,)))

# step 3 - Flatten
classifier.add(Flatten())

#step 4 - ful connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# compling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# part 2 - fitting cnn to image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('D:\\Programming\\python\\cat_vs_dog\\train1',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = train_datagen.flow_from_directory('D:\\Programming\\python\\cat_vs_dog\\test1',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 1, 
                         validation_data = test_set,
                         validation_steps = 1)

# part 3 - making new prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('D:\Programming\python\cat_vs_dog\cat.jpg', 
                            target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
