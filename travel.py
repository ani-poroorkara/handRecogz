import os 
import numpy as np 

x_data = []
y_data = []
datacount = 0

for i in range (0, 15): 
    if i < 10:
        for j in os.listdir('dataset/near-infrared/0' + str(i) + '/' + 'train_pose'+ '/'):
            if not j.startswith('.'):
                for k in os.listdir('dataset/near-infrared/0' + str(i) + '/' + 'train_pose'+ '/' + j + '/'):
                    datacount = datacount+1
                label = j[:2]   
                label = int(label)
                print("L : " + str(label))
    else: 
        for j in os.listdir('dataset/near-infrared/' + str(i) + '/' + 'train_pose'+ '/'):
            if not j.startswith('.'):
                for k in os.listdir('dataset/near-infrared/' + str(i) + '/' + 'train_pose'+ '/' + j + '/'):
                    datacount = datacount+1
                label = j[:2]   
                label = int(label)
                print("L : " + str(label))