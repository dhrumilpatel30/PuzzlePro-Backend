import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from  sklearn.model_selection import  train_test_split
from keras.preprocessing.image import ImageDataGenerator



################################################
path='Digits'

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

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(img_train)

print(len(img_train))


#plot figures

#distribution of images

# numOfSamples= []
# for x in range(0,noOfClasses):
#     numOfSamples.append(len(np.where(x_train==x)[0]))
# print(numOfSamples)
# plt.figure(figsize=(10,10))
# plt.bar(range(0,noOfClasses),numOfSamples)
# plt.title("No of Images for each Class")
# plt.xlabel("Class ID")
# plt.ylabel("Number of Images")
# plt.xticks(range(10))
# plt.show()
#
#

