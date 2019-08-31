# coding: utf-8

from PIL import Image
import numpy as np
import pickle
import csv
def Normalize(image,mean,std):
    for channel in range(3):
        image[:,:,channel]=(image[:,:,channel]-mean[channel])/std[channel]
    return image
size_of_file=5000
size_of_file2=10
id_to_data={}
id_to_size={}
with open("./data/training.csv") as f:
    csv_reader=csv.reader(f)
    counter=0
    for line in csv_reader:
        if counter==0:
            counter=counter+1
            continue
        id=counter
        path=line[0]
        image=Image.open("./data/images/trainingimage/"+path).convert('RGB')
        id_to_size[int(id)]=np.array(image,dtype=np.float32).shape[0:2]
        image=image.resize((224,224))
        image=np.array(image,dtype=np.float32)
        image=image/255
        image=Normalize(image,[0.485,0.456,0.406],[0.229,0.224,0.225])
        id_to_data[int(id)]=image
        if counter%1000==0:
        	print(counter)
        #if counter>=size_of_file:
        	#break
        counter=counter+1

  
        
id_to_data=np.array(list(id_to_data.values()))
id_to_size=np.array(list(id_to_size.values()))
f=open("./id_to_data_training","wb+")
pickle.dump(id_to_data,f,protocol=4)
f=open("./id_to_size_training","wb+")
pickle.dump(id_to_size,f,protocol=4)


id_to_box={}  
with open("./data/training.csv") as f:
    csv_reader=csv.reader(f)
    counter=0
    for line in csv_reader:
        if counter==0:
            counter=counter+1
            continue
        id=counter
        box=[float(line[1]),float(line[2]),float(line[3]),float(line[4])]
        box[0]=(box[0]/id_to_size[int(id)-1][1])*224
        box[1]=(box[1]/id_to_size[int(id)-1][1])*224
        box[2]=(box[2]/id_to_size[int(id)-1][0])*224
        box[3]=(box[3]/id_to_size[int(id)-1][0])*224
        id_to_box[int(id)]=box
        #if counter>=size_of_file:
        	#break
        counter=counter+1
 
id_to_box=np.array(list(id_to_box.values()))
#print(id_to_box)
f=open("./id_to_box_training","wb+") 
pickle.dump(id_to_box,f,protocol=4)





id_to_data={}
id_to_size={}
with open("./data/test1.csv") as f:
    csv_reader=csv.reader(f)
    counter=0
    for line in csv_reader:
        if counter==0:
            counter=counter+1
            continue        
        id=counter
        path=line[0]
        #print(path)
        image=Image.open("./data/images/testimage1/"+path).convert('RGB')
        id_to_size[int(id)]=np.array(image,dtype=np.float32).shape[0:2]
        image=image.resize((224,224))
        image=np.array(image,dtype=np.float32)
        image=image/255
        image=Normalize(image,[0.485,0.456,0.406],[0.229,0.224,0.225])
        id_to_data[int(id)]=image
        #if counter >=size_of_file2:
            #break
        counter=counter+1    
        
id_to_data=np.array(list(id_to_data.values()))
id_to_size=np.array(list(id_to_size.values()))
f=open("./id_to_data_test","wb+")
pickle.dump(id_to_data,f,protocol=4)
f=open("./id_to_size_test","wb+")
pickle.dump(id_to_size,f,protocol=4)






id_to_box={}  
with open("./data/test1.csv") as f:
    csv_reader=csv.reader(f)
    counter=0
    for line in csv_reader:
        if counter==0:
            counter=counter+1
            continue        
        id=counter
        box=[0.0,0.0,0.0,0.0,line[0] ]
        id_to_box[int(id)]=box
        #if counter >=size_of_file2:
            #break
        counter=counter+1    

id_to_box=np.array(list(id_to_box.values()))
f=open("./id_to_box_test","wb+") 
pickle.dump(id_to_box,f,protocol=4)
f=open("./id_to_box_test","rb+") 
print(pickle.load(f)[0])
