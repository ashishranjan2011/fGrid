import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import random
import tensorflow as tf
import csv
def my_metric(labels,predictions):
    threshhold=0.75
    x=predictions[:,0]*224
    x=tf.maximum(tf.minimum(x,224.0),0.0)
    y=predictions[:,1]*224
    y=tf.maximum(tf.minimum(y,224.0),0.0)
    width=predictions[:,2]*224
    width=tf.maximum(tf.minimum(width,224.0),0.0)
    height=predictions[:,3]*224
    height=tf.maximum(tf.minimum(height,224.0),0.0)
    label_x=labels[:,0]
    label_y=labels[:,1]
    label_width=labels[:,2]
    label_height=labels[:,3]
    a1=tf.multiply(width,height)
    a2=tf.multiply(label_width,label_height)
    x1=tf.maximum(x,label_x)
    y1=tf.maximum(y,label_y)
    x2=tf.minimum(x+width,label_x+label_width)
    y2=tf.minimum(y+height,label_y+label_height)
    IoU=tf.abs(tf.multiply((x1-x2),(y1-y2)))/(a1+a2-tf.abs(tf.multiply((x1-x2),(y1-y2))))
    condition=tf.less(threshhold,IoU)
    sum=tf.where(condition,tf.ones(tf.shape(condition)),tf.zeros(tf.shape(condition)))
    return tf.reduce_mean(sum)

def smooth_l1_loss(true_box,pred_box):
    loss=0.0
    for i in range(4):
        residual=tf.abs(true_box[:,i]-pred_box[:,i]*224)
        condition=tf.less(residual,1.0)
        small_res=0.5*tf.square(residual)
        large_res=residual-0.5
        loss=loss+tf.where(condition,small_res,large_res)
    return tf.reduce_mean(loss)


plt.switch_backend('agg')
size_of_file = 12816
f=open("./id_to_data_test","rb+")
data=pickle.load(f)

f=open("./id_to_box_test","rb+")
box=pickle.load(f)

f=open("./id_to_size_test","rb+") 
size=pickle.load(f)

index=[i for i in range(size_of_file)]
#index=random.sample(index,8)


model=keras.models.load_model('./model.h5', custom_objects={'smooth_l1_loss':smooth_l1_loss,'my_metric':my_metric})
result=model.predict(data[index,:,:,:])

a = result
with open("output.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(a)

mean=[0.485,0.456,0.406]
std=[0.229,0.224,0.225]
j=0

#for outputing coordinates in csv file
my_list=[]
with open("./data/test.csv") as f:
    csv_reader=csv.reader(f)
    counter=0
    for line in csv_reader:
        my_list.append([line[0]])
        counter=counter+1



my_list[0].append('x1')
my_list[0].append('x2')
my_list[0].append('y1')
my_list[0].append('y2')
list_index=1
for row in result:
    count=0;
    for cordinate in row:
    	if count%4==0 or count%4==1:
    		cordinate=cordinate*640
    	else :
    		cordinate=cordinate*480
        my_list[list_index].append(cordinate)
    list_index=list_index+1

with open("./data/my_output.csv","w+") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in my_list:
        writer.writerow(val)  

