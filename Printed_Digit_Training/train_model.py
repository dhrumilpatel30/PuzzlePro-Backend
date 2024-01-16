import cv2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time
import keras
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical


def normalize_image(current_image):
    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    current_image = cv2.equalizeHist(current_image)
    current_image = current_image / 255
    return current_image


def train_printed_digit_model(train_images, train_labels, number_of_classes):
    batch_size_value = 50
    epochs_value = 20

    train_images = np.array(list(map(normalize_image, train_images)))

    train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)

    train_labels = to_categorical(train_labels, number_of_classes)

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    datagen.fit(train_images)

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', patience=3, verbose=1,
                                                                factor=0.5, min_lr=0.00001)
    model.fit(train_images, train_labels, batch_size=batch_size_value, epochs=epochs_value, verbose=2,
              callbacks=[learning_rate_reduction])

    start_time = time.time()

    print("---Training time %s seconds ---" % (time.time() - start_time))

    model.save("../Models/printed_digits_model.keras")
