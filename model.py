#Import libraries
import numpy as np
import os 
from PIL import Image 
import keras
from keras.utils import to_categorical
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
import cv2

#Create a list for labels and data
x_data = []
y_data = []

#Initialize data count to zero
datacount = 0

#Cycle through the folders and then images
for i in range (0, 15): 
    if i < 10:
        for j in os.listdir('dataset/near-infrared/0' + str(i) + '/' + 'train_pose'+ '/'):
            if not j.startswith('.'):
                for k in os.listdir('dataset/near-infrared/0' + str(i) + '/' + 'train_pose'+ '/' + j + '/'):
                    img = Image.open('dataset/near-infrared/0' + str(i) + '/' + 'train_pose'+ '/' + j + '/'+ k)
                    img = img.resize((56,56))
                    img = np.array(img)
                    img = cv2.GaussianBlur(img, (5, 5), 0) 
                    thresholded = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)[1]
                    x_data.append(thresholded)
                    label = j[:2]   
                    label = int(label)
                    y_data.append(label)
                    datacount = datacount+1
                    
    else: 
        for j in os.listdir('dataset/near-infrared/' + str(i) + '/' + 'train_pose'+ '/'):
            if not j.startswith('.'):
                for k in os.listdir('dataset/near-infrared/' + str(i) + '/' + 'train_pose'+ '/' + j + '/'):
                    img = Image.open('dataset/near-infrared/' + str(i) + '/' + 'train_pose'+ '/' + j + '/'+ k)
                    img = img.resize((56,56))
                    img = np.array(img)
                    img = cv2.GaussianBlur(img, (5, 5), 0) 
                    thresholded = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)[1]
                    x_data.append(thresholded)
                    label = j[:2]   
                    label = int(label)
                    y_data.append(label)
                    datacount = datacount+1

#Convert the acquired data to numpy array
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)

np.save('x_data', x_data)
np.save('y_data', y_data)

print("Total training data:" + str(datacount))

#Repeat the same with testing data 
x_data_test = []
y_data_test = []
datacount_test = 0

for i in range (0, 15): 
    if i < 10:
        for j in os.listdir('dataset/near-infrared/0' + str(i) + '/' + 'test_pose'+ '/'):
            if not j.startswith('.'):
                for k in os.listdir('dataset/near-infrared/0' + str(i) + '/' + 'test_pose'+ '/' + j + '/'):
                    img = Image.open('dataset/near-infrared/0' + str(i) + '/' + 'test_pose'+ '/' + j + '/'+ k)
                    img = img.resize((56,56))
                    img = np.array(img)
                    img = cv2.GaussianBlur(img, (5, 5), 0) 
                    thresholded = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)[1]
                    x_data_test.append(thresholded)
                    label = j[:2]   
                    label = int(label)
                    y_data_test.append(label)
                    datacount_test = datacount_test +1
                    print(k)

                    
    else: 
        for j in os.listdir('dataset/near-infrared/' + str(i) + '/' + 'test_pose'+ '/'):
            if not j.startswith('.'):
                for k in os.listdir('dataset/near-infrared/' + str(i) + '/' + 'test_pose'+ '/' + j + '/'):
                    img = Image.open('dataset/near-infrared/' + str(i) + '/' + 'test_pose'+ '/' + j + '/'+ k)
                    img = img.resize((56,56))
                    img = np.array(img)
                    img = cv2.GaussianBlur(img, (5, 5), 0) 
                    thresholded = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)[1]
                    x_data_test.append(thresholded)
                    label = j[:2]   
                    label = int(label)
                    y_data_test.append(label)
                    datacount_test = datacount_test +1
                    print(k)

x_data_test = np.array(x_data_test, dtype = 'float32')
y_data_test = np.array(y_data_test)

np.save('x_data_test', x_data_test)
np.save('y_data_test', y_data_test)

print("Total testing data: " + str(datacount_test))


#Reshape and convert y data to labels 
y_data = to_categorical(y_data)
x_data = x_data.reshape((datacount, 56, 56, 1))
x_data /= 255

y_data_test = to_categorical(y_data_test)
x_data_test = x_data_test.reshape((datacount_test, 56, 56, 1))
x_data_test /= 255


#Create the model 
model=models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(56, 56,1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(19, activation='softmax'))

#Compile the model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_data, y_data, epochs=20, batch_size=64, verbose=1, validation_data=(x_data_test, y_data_test))
[loss, acc] = model.evaluate(x_test,y_test,verbose=1)
print("Accuracy:" + str(acc))

#Save model
model.save('gesture.h5')