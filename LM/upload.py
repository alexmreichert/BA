
#############
#############
"""imports"""
#############
#############

#azure
from datetime import datetime
from azure.storage.blob import ContainerClient
from azure.core import exceptions
import numpy as np
import yaml
from azure.servicebus import ServiceBusClient, ServiceBusMessage
import os
import time


#azure
def load_config(dir_config):
    with open(os.path.join(dir_config + r"\config.yaml"), "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)

def upload(files, connection_string, container_name, servicebus_client):

    container_client = ContainerClient.from_connection_string(connection_string, container_name)   
    if not container_client.exists():
        log(servicebus_client,f"creating new containter with {container_name}")
        try:
            container_client.create_container()
        except exceptions.ResourceExistsError:
            log(servicebus_client,"waiting for deletion")
            time.sleep(20)
            container_client.create_container()

    log(servicebus_client,"uploading files:")     
    for file in files:
        blob_client = container_client.get_blob_client(file.name)
        with open(file.path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
    log(servicebus_client,f"uploaded {files}")

def send_message(servicebus_client,queue_name, message):
    sender = servicebus_client.get_queue_sender(queue_name=queue_name)
    with sender:  
          #'done--{container_name}--{Bauteil-19.stl}--{True}--{changestring}'
          #changestring=r_n_s_v_slicevariation_mode         else none  
        sender.send_messages(ServiceBusMessage(message))
        log(servicebus_client,f"sent message: {message}")

def get_files(dir):
    with os.scandir(dir) as entries:
        for entry in entries:
            if (entry.is_file()) and ((entry.name.endswith(".stl") or entry.name.endswith(".ply"))):
                yield entry # turns into a generator function

def log(service_bus_client,log,sender_info="LM-"):
    sender = service_bus_client.get_queue_sender(queue_name="log")
    message=sender_info+str(datetime.now())+": "+log
    with sender:  
        sender.send_messages(ServiceBusMessage(message))
        print("log: %s"%(message))

#############
#############
"""execute"""
#############
#############
current_dir=os.path.dirname(os.path.abspath(__file__))
config=load_config(current_dir)
servicebus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connectionstring'])


pipename_local_to_data_prep = "lmdp"
pipename_data_prep_to_local= "dplm"
pipename_data_prep_to_ID_NN= "dpin"
pipename_data_prep_to_ID_reg= "dpir"
pipename_img_gen_to_train= "imtr"
pipename_local_to_img_gen= "lmim"


variation_NN={
    "radius":[1,4],
    "noise": [4,16],
    "view": [1,8],
    "batch": [16,64],
    "size": [2048,2048],
}

variation_reg={
    "points":[1024,4096],
    "radius": [0.01,0.02],
    "feature": [0.004,0.008],
    "noise": [0.0001,0.0004],
}

#'done--{container_name}--{label}--{config}--{changestr}'
#done--containerlmim--Bauteil-013.stl--True--2.2_10
# queue_name=config["queue_name"]+pipename_data_prep_to_ID_reg
# container_name=config["container_name"]+pipename_data_prep_to_ID_reg
# label="part3.pcd"
# message=f'done--{container_name}--{label}--False--'
# send_message(servicebus_client,queue_name,message)
# quit()


variation=variation_reg
length_variation=len(variation_reg)
mode="reg"
ml=True
if ml:
    mode="ml"
    length_variation=len(variation_NN)
    variation=variation_NN

#get variations
variations=np.zeros((2**length_variation,length_variation))
for num in range(2**length_variation):
    binary=np.asarray(list(str(bin(num))[2:]), dtype=int)
    row=binary
    if len(binary)<length_variation:
        row_add=np.zeros((1, length_variation-len(binary)), dtype=int).flatten()
        row=np.hstack((row_add,binary))
    variations[num]=row
# print(variations)
# loading stl files
container_name=config['container_name']+pipename_local_to_img_gen
queue_name=config['queue_name']+pipename_local_to_img_gen
# files=get_files(os.path.join(current_dir, config['source_folder']))
# upload(files,config['azure_storage_connectionstring'],container_name,servicebus_client)

keys=list(variation.keys())
# variations=np.zeros((1,length_variation))


# print(variations)
# send out configstr
row_i=0
# skip_to=15
iter=0

for row in variations:
    row_i+=1
    # if row_i<skip_to: continue
    configstr=mode+"_"
    for i in range(len(row)):
        configstr+=keys[i]+"-"+str(int(row[i]))+"_"
        # variation[keys[i]][int(row[i])]
    
    if row_i in [16]: #8
        
        configstr+="variation-%s_"%(41+iter)
        iter+=1
        send_message(servicebus_client,queue_name,configstr)
        # print(configstr)
        # break