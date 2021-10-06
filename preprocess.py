import numpy as np
import os
from PIL import Image

#Data Preprocessing 
lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('dataset/00/'):
    if not j.startswith('.'):
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1

x_data = []
y_data = []
datacount = 0 
for i in range(0, 10): 
    for j in os.listdir('dataset/0' + str(i) + '/'):
        if not j.startswith('.'): 
            count = 0 
            for k in os.listdir('dataset/0' + str(i) + '/' + j + '/'):
                img = Image.open('dataset/0' + str(i) + '/' + j + '/' + k)
                img_grey = img.convert('L')
                value = np.asarray(img_grey.getdata(), dtype=np.int).reshape(512,300)
                value = value.flatten()
                x_data.append(value) 
                count = count + 1
            y_values = np.full((count, 1), lookup[j]) 
            y_data.append(y_values)
            datacount = datacount + count
np.savetxt("datacsv.csv", x_data)
np.savetxt("label.csv", y_data)
y_data = y_data.reshape(datacount, 1) 