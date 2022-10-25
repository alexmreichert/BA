import os
from tqdm import tqdm
from azure.storage.blob import ContainerClient
from azure.core import exceptions
import yaml
from distutils.command.config import config
from tkinter import N
from azure.servicebus import ServiceBusClient, ServiceBusMessage, exceptions 

import open3d as o3d
import numpy as np
import h5py

#functions
def load_config(dir_config):
    with open(os.path.join(dir_config, "config.yaml"), "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)



pipename_local_to_data_prep = "lmdp"
pipename_data_prep_to_local= "dplm"
pipename_data_prep_to_ID_NN= "dpin"
pipename_data_prep_to_ID_reg= "dpir"
pipename_img_gen_to_train_reg= "imtr"
pipename_img_gen_to_train_ML= "imtm"
pipename_train_ML_to_train_ML= "tmtm"
pipename_local_to_img_gen= "lmim"

point_length=2048
test_portion=0.2
config=load_config("")

# #get points from container 
servicebus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connectionstring'])
filenames=[]
with servicebus_client:
    queue_name=  "queue"+pipename_img_gen_to_train_ML
    receiver = servicebus_client.get_queue_receiver(queue_name=queue_name, max_wait_time=30)
    # receiver = servicebus_client.get_queue_receiver(queue_name=queue_name, sub_queue="deadletter")
    with receiver:
        messages = receiver.receive_messages(max_message_count=100000,max_wait_time=30)
        # print(f"messages: {len(messages)}")  
        for message in tqdm(messages):
            if str(message).startswith("done"): filenames.append(str(message).split("--")[2])
            else: print(f"Wierd message: {message}")
            receiver.complete_message(message)
            # print(str(message).split("--")[2])
        print(f"cleared {queue_name}")

#get labels

size_dataset=len(filenames)
labels_to_prune=np.zeros((size_dataset))
label_names=[]#['Bauteil-007', 'Bauteil-013', 'Bauteil-020', 'Bauteil-025', 'Bauteil-030']#[]
i=0
for names in filenames:
    label=names.split("_")[0]
    if not label in label_names: label_names.append(label)
    labels_to_prune[i]=label_names.where(label)
print(label_names)
print(f"len filenames: {size_dataset}")

#prune to equalize 
unique, counts = np.unique(labels_to_prune, return_counts=True)
print(f"unique, counts: {unique, counts}")
prune_len=np.amin(counts)
print(prune_len)


#initiate nparray for points and labels
points=np.zeros((size_dataset,point_length,3)) #shape: size_dataset x point_lenght x 3
labels=np.zeros((size_dataset)) #shape: size_dataset

#grab points and labels 
container_name=config["container_name"]+pipename_img_gen_to_train_ML
container_client = ContainerClient.from_connection_string(config["azure_storage_connectionstring"], container_name)
save_path=config["save_folder"]
if not container_client.exists(): 
    print("wrong connection_string")
else:
    blobs=container_client.list_blobs()
    # print(f"len blobs {len(blobs)}")
    counter=0
    for blob in blobs:
        blob_name=blob.name
        if blob_name in filenames:
            
            StorageStreamDownloader = container_client.download_blob(blob)
            filename=os.path.join(save_path,blob_name)
            try:
                file = open(filename, 'wb')
            except FileNotFoundError:
                os.mkdir(save_path)
                file = open(filename, 'wb')
            data=StorageStreamDownloader.readall()
            # container_client.delete_blob(blob, delete_snapshots="include")
            file.write(data)
            file.close()

            pcd=o3d.io.read_point_cloud(filename)
            pcd_points=np.asarray(pcd.points)
            idx=np.arange(len(pcd_points))
            np.random.shuffle(idx)
            idx = idx[:point_length]
            points_sampled=pcd_points[idx,:]
            points[counter,:,:]=points_sampled
            blob_label=blob_name.split("_")[0]
            labels[counter]=label_names.index(blob_label)
            counter+=1
            os.remove(filename)
        else: print(f"NOOOOOOOOOO: {blob_name}")  
        # container_client.delete_blob(blob, delete_snapshots="include")
    # print(f"cleared {container_name}")

##split dataset into test and trian
idx_dataset=np.arange(size_dataset)#idx 
np.random.shuffle(idx_dataset)#shuffle
idx_cutoff=int(np.ceil((size_dataset*(1-test_portion))))#idx_of_idx  cut off
idx_train=idx_dataset[:idx_cutoff]#idx for testing
idx_test=idx_dataset[idx_cutoff:]#idx for training
train_points=points[idx_train,:,:]#train points
test_points=points[idx_test,:,:]#test points#
train_labels=labels[idx_train]#train labels
test_labels=labels[idx_test]#test labels
#create h5py file
print(f"points: {train_points.shape}")
print(f"points: {test_points.shape}")
print(f"labels: {train_labels.shape}")
print(f"labels: {test_labels.shape}")

f = h5py.File('10_2048data_train.h5', 'w')
# group=f.create_group("data")
dataset_points=f.create_dataset(name="data",shape=train_points.shape,dtype=np.dtype("float32"),data=train_points)
# dataset_points=points
dataset_points=f.create_dataset(name="label",shape=train_labels.shape,dtype=np.dtype(int),data=train_labels)
# dataset_points=labels
f.close()

f = h5py.File('10_2048data_test.h5', 'w')
# group=f.create_group("data")
dataset_points=f.create_dataset(name="data",shape=test_points.shape,dtype=np.dtype("float32"),data=test_points)
# dataset_points=points
dataset_points=f.create_dataset(name="label",shape=test_labels.shape,dtype=np.dtype(int),data=test_labels)
# dataset_points=labels
f.close()
