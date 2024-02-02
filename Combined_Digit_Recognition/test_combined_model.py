import keras.models
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools


def test_combined_digit_model(test_images, test_labels):
    model = keras.models.load_model('C:/Users/user1/digitmodel/PuzzlePro-Backend/Models/combined_digit_model.keras')

    start_time = time.time()
    predictions = model.predict(test_images)
    print("---Testing time %s seconds ---" % (time.time() - start_time))

    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # noinspection PyUnresolvedReferences
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

            if not normalize:
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
    print(classification_report(test_labels.argmax(axis=1), predictions.argmax(axis=1)))
