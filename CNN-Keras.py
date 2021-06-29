from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
#Max pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#3rd Conv layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
#Max pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
#output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('C:\\Users\\Asus-2020\Downloads\\DS\\train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('C:\\Users\\Asus-2020\Downloads\\DS\\test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
#Steps per epoch - no of iterations- 1 million datas - 1million/2000
#Validation_data - if have any specific validation data then you can choose but here i choose test data as validation data
#validataion_steps - After 1500 steps do the validation
classifier.fit(training_set,
                         steps_per_epoch = 8000,
                         epochs = 15,
                         validation_data = test_set,
                         validation_steps = 2000)

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:\\Users\\Asus-2020\Downloads\\Dhoni11.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Dhoni'
    print(prediction)
else:
    prediction = 'Sachin'
    print(prediction)