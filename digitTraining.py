import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
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



img_train = np.array(list(map(preProcessing,img_train)))
img_test = np.array(list(map(preProcessing,img_test)))
print(img_train.shape)
print(img_test.shape)

#image reshape
img_train = img_train.reshape(img_train.shape[0],img_train.shape[1],img_train.shape[2],1)
img_test = img_test.reshape(img_test.shape[0],img_test.shape[1],img_test.shape[2],1)


#distribution of images

numOfSamples= []
for x in range(0,noOfClasses):
    numOfSamples.append(len(np.where(x_train==x)[0]))
print(numOfSamples)
plt.figure(figsize=(10,10))
plt.bar(range(0,noOfClasses),numOfSamples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.xticks(range(10))
plt.show()


dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(img_train)

print(len(img_train))

#### IMAGE AUGMENTATION
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(img_train)


x_train = to_categorical(x_train,noOfClasses)
x_test = to_categorical(x_test,noOfClasses)

#### CREATING THE MODEL
def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add(Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu'))
    model.add(Conv2D(noOfFilters, sizeOfFilter1, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu'))
    model.add(Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# model = myModel()
# print(model.summary())

# #  training the model
# history = model.fit(dataGen.flow(img_train,x_train, batch_size=batchSizeVal),
#                               epochs = epochsVal,
#                               verbose = 2,
#                               steps_per_epoch=img_train.shape[0],
#                               )

#