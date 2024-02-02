import os
import time
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization


def create_model(input_shape, number_of_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))

    model.add(Dense(number_of_classes, activation="softmax"))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_combined_digit_model(train_images, test_images, train_labels, test_labels, number_of_classes):
    batch_size = 50
    epochs = 100

    input_shape = train_images.shape[1:]
    model = create_model(input_shape, number_of_classes)

    save_dir = '../Models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Training Log\n")
    print("-------------\n\n")

    print("Model Summary:")
    model.summary(print_fn=lambda x: print(x))

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    datagen.fit(train_images)

    start_time = time.time()

    learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', patience=3, verbose=1,
                                                factor=0.5, min_lr=0.00001)

    model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
              steps_per_epoch=len(train_images) // batch_size,
              epochs=epochs,
              validation_data=(test_images, test_labels), callbacks=[learning_rate_reduction])

    print("---Training time %s seconds ---" % (time.time() - start_time))

    print("\n\nAdditional Details:")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Input Shape: {input_shape}")
    print(f"Number of Classes: {number_of_classes}")

    model.save("")
    print("Model saved at:", '')
