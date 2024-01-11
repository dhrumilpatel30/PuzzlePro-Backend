import keras.callbacks
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools


sns.set(style='white', context='notebook', palette='deep')


################################################
path='Digits'
imageDimensions= (32,32,3)
batchSizeVal= 50
epochsVal = 2
stepsPerEpochVal = 2000

################################################


# data preprocessing for training and testing
images =[]
classNo =[]

List = os.listdir(path)
print("Total classes detected :", len(List))
noOfClasses = len(List)

print("Importing classes.....")

for x in range(0,noOfClasses):
    PicList = os.listdir(
        path+"/"+ str(x)
    )
    for y in PicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        images.append(curImg)
        classNo.append(x)
    print(x,end=" ")
print(" ")

print("Total images in imageList : ",len(images))
print("Total classes in classLabelList : ",len(classNo))

#convert to numpy arrays
images = np.array(images)
classNo = np.array(classNo)
print(images.shape)

# splitting the data
img_train,img_test,x_train,x_test = train_test_split(images,classNo,train_size=0.80,test_size=0.20)
print(img_train.shape)
print(img_test.shape)
print(x_train.shape)
print(x_test.shape)

# free some space
del images,classNo

# preprocessing function for images
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


# preprocessing the training images
img_train = np.array(list(map(preProcessing,img_train)))
img_test = np.array(list(map(preProcessing,img_test)))

img_train = img_train.reshape(img_train.shape[0],img_train.shape[1],img_train.shape[2],1)
img_test = img_test.reshape(img_test.shape[0],img_test.shape[1],img_test.shape[2],1)

# checking the correctness
# print(img_train.shape)
# cv2.imshow("GrayScale Image",img_train[10])
# print(x_train[10])
# cv2.waitKey(0)

x_train = to_categorical(x_train,noOfClasses)
x_test = to_categorical(x_test,noOfClasses)



# creating the model
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(img_train, x_train, batch_size = batchSizeVal, epochs = 18,verbose = 2)

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

#calculating training and testing time
print("---Training time %s seconds ---" % (time.time() - start_time))

start_time = time.time()
predictions = model.predict(img_test)
print("---Testing time %s seconds ---" % (time.time() - start_time))

score = model.evaluate(img_test, x_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#calculating confusion matrix
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

plot_confusion_matrix(confusion_matrix(x_test.argmax(axis=1), predictions.argmax(axis=1)), range(10))

#calculating Precision, Recall and F1 score
print(classification_report(x_test.argmax(axis=1), predictions.argmax(axis=1)))


cv2.imshow("GrayScale Image",img_test[10])
print(x_test[10])
cv2.waitKey(0)
ans =model.predict(img_test[10])
print(ans)



import pickle

# Assuming 'model' is your trained model object
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)



print("complete")


