import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau


def custom_print(*args, file=None, flush=False):
    print(*args, flush=flush)

    if file is not None:
        with open(file, 'a') as log_file:
            print(*args, file=log_file, flush=flush)


def create_model(input_shape, number_of_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_combined_digit_model(train_images, test_images, train_labels, test_labels, number_of_classes):
    batch_size = 50
    epochs = 50

    input_shape = train_images.shape[1:]
    model = create_model(input_shape, number_of_classes)

    save_dir = '../Models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file_path = os.path.join(save_dir, 'combined_digit_model_training_log.txt')

    with open(log_file_path, 'w') as log_file:
        log_file.write("Training Log\n")
        log_file.write("-------------\n\n")

        custom_print("Model Summary:", file=log_file)
        model.summary(print_fn=lambda x: custom_print(x, file=log_file))

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

        history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
                            steps_per_epoch=len(train_images) // batch_size,
                            epochs=epochs,
                            validation_data=(test_images, test_labels), callbacks=[learning_rate_reduction])

        custom_print("---Training time %s seconds ---" % (time.time() - start_time), file=log_file)

        custom_print("\n\nTraining History:", file=log_file)
        log_file.write("Epoch\t")
        for key in history.history.keys():
            log_file.write(f"{key}\t")
        log_file.write("\n")

        for epoch in range(epochs):
            custom_print(f"\nEpoch {epoch + 1}/{epochs}", file=log_file, flush=True)
            for step, (x_batch, y_batch) in enumerate(datagen.flow(train_images, train_labels, batch_size=batch_size)):
                model.train_on_batch(x_batch, y_batch)
                custom_print(f"\rTraining: Step {step + 1}/{len(train_images) // batch_size}")

        custom_print("\n\nAdditional Details:", file=log_file)
        custom_print(f"Batch Size: {batch_size}", file=log_file)
        custom_print(f"Epochs: {epochs}", file=log_file)
        custom_print(f"Input Shape: {input_shape}", file=log_file)
        custom_print(f"Number of Classes: {number_of_classes}", file=log_file)

    custom_print("Training log saved at:", log_file_path)

    model.save(os.path.join(save_dir, 'combined_digit_model.keras'))
    custom_print("Model saved at:", os.path.join(save_dir, 'combined_digit_model.keras'))