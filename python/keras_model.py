import time

from keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPool2D
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu', input_shape=(32, 32, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'GeneratedDB\\train',
        target_size=(32, 32),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'GeneratedDB\\test',
        target_size=(32, 32),
        batch_size=600,
        color_mode='grayscale',
        class_mode='categorical')

cpt = 0
for i in range(60):
    train = model.fit_generator(
            train_generator,
            epochs=1,
            validation_data=validation_generator)
    time.sleep(1.5)
    if train.history['accuracy'] > train.history['val_accuracy'] :
        cpt += 1
        if cpt == 4:
            break
model.save("DeepLens_sudoku_model3.hdf5")