from Printed_Digit_Training.load_datasets import load_printed_digit_dataset
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# from keras.preprocessing.image import ImageDateGe
from sklearn.metrics import confusion_matrix
import itertools


def train_printed_digit_model():

    all_images, class_number = load_printed_digit_dataset()
    number_of_classes = len(np.unique(class_number))

    train_images, test_images, train_labels, test_labels = train_test_split(all_images, class_number, train_size=0.80,
                                                                            test_size=0.20)

    del all_images, class_number

    def normalize_image(current_image):
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        current_image = cv2.equalizeHist(current_image)
        current_image = current_image / 255
        return current_image

    train_images = np.array(list(map(normalize_image, train_images)))
    test_images = np.array(list(map(normalize_image, test_images)))

    train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)

    train_labels = to_categorical(train_labels, number_of_classes)
    test_labels = to_categorical(test_labels, number_of_classes)

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

    history = model.fit(train_images, train_labels, batch_size=batchSizeVal, epochs=18, verbose=2)

    # #data augmentation
    # datagen = ImageDataGenerator(
    #         featurewise_center=False,  # set input mean to 0 over the dataset
    #         samplewise_center=False,  # set each sample mean to 0
    #         featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #         samplewise_std_normalization=False,  # divide each input by its std
    #         zca_whitening=False,  # apply ZCA whitening
    #         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    #         zoom_range = 0.1, # Randomly zoom image
    #         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #         horizontal_flip=False,  # randomly flip images
    #         vertical_flip=False)  # randomly flip images
    #
    #
    # datagen.fit(img_train)
    #
    # learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='accuracy',
    #                                             patience=3,
    #                                             verbose=1,
    #                                             factor=0.5,
    #                                             min_lr=0.00001)
    #
    start_time = time.time()
    # # Fit the model
    # history = model.fit(datagen.flow(img_train,x_train, batch_size=batchSizeVal),
    #                                   epochs = 20, verbose = 2,callbacks=[learning_rate_reduction])

    # calculating training and testing time
    print("---Training time %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    predictions = model.predict(test_images)
    print("---Testing time %s seconds ---" % (time.time() - start_time))

    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # calculating confusion matrix
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):

        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=20)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
        plt.yticks(tick_marks, classes, fontsize=15)

        if normalize:
            cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)

        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            if normalize == False:
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black", fontsize=20)
            else:
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black", fontsize=20)

        plt.tight_layout()
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)
        plt.show()

    plot_confusion_matrix(confusion_matrix(test_labels.argmax(axis=1), predictions.argmax(axis=1)), range(10))

    # calculating Precision, Recall and F1 score
    print(classification_report(test_labels.argmax(axis=1), predictions.argmax(axis=1)))

    cv2.imshow("GrayScale Image", test_images[10])
    print(test_labels[10])
    cv2.waitKey(0)
    ans = model.predict(test_images[10])
    print(ans)

    import pickle

    # Assuming 'model' is your trained model object
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print("complete")


