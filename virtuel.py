"""imports"""
from distutils.command.config import config
from tkinter import N
from azure.servicebus import ServiceBusClient, ServiceBusMessage
import os
import yaml
import time
from azure.storage.blob import ContainerClient
from azure.core import exceptions
import re

"""functions"""
#create data table
def splicename(file_name):
    match = re.match(r"\w+(\d+)\w\w(\d+)(\.\w+)", file_name)
    if match:
        n=match.group(1)
        N=match.group(2)
        file_type=match.group(3)
        return n, N, file_type
    else: 
        print(f"file name: {file_name} in wrong format use \w+(d+)ww(d+)(\.\w+)")
        return 0, 0, "error"
#download 
def download(connection_string, container_name, save_path):
    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    if not container_client.exists(): 
        print("wrong connection_string")
        return None
    else:
        blobs=container_client.list_blobs()
        for blob in blobs:
            StorageStreamDownloader = container_client.download_blob(blob)
            filename=os.path.join(save_path,blob.name)
            #print(StorageStreamDownloader)
            try:
                file = open(filename, 'wb')
            except FileNotFoundError:
                os.mkdir(save_path)
                file = open(filename, 'wb')
            data=StorageStreamDownloader.readall()
            print(f"saving localy and deleting blob: {blob.name}")
            container_client.delete_blob(blob, delete_snapshots="include")
            file.write(data)
            file.close()

def getPLY(config, servicebus_client):
    #wait for completion message and download file
    with servicebus_client:
        # get the Subscription Receiver object for the subscription    
        receiver = servicebus_client.get_queue_receiver(queue_name=config['queue_name'], max_wait_time=30)
        with receiver:
            messages = receiver.receive_messages(max_wait_time=1)
            i=1
            while ((not messages) or str(messages[0])[:4]!="done") and i<40: 
                print("waiting for upload completion")
                time.sleep(1)
                messages = receiver.receive_messages(max_wait_time=1)
                i+=1
            if messages:
                for message in messages:
                    strmsg=str(message)
                    if(strmsg[:4]=="done"):
                        print(f"starting download from container {strmsg[5:]}")
                        download(config['azure_storage_connectionstring'],strmsg[5:], config['save_folder'])
                    receiver.complete_message(message)

#savefile
def load_config(dir_config):
    with open(os.path.join(dir_config + r"\config.yaml"), "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)

def upload(files, connection_string, container_name):

    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    if not container_client.exists():
        print(f"creating new containter with {container_name}")
        try:
            container_client.create_container()
        except exceptions.ResourceExistsError:
            print("waiting for deletion")
            time.sleep(20)
            container_client.create_container()

    print("uploading files:")        
    for file in files:
        blob_client = container_client.get_blob_client(file.name)
        with open(file.path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
            print(f"{file.name} uploaded")

def send_completion_message(servicebus_client,queue_name, container_name):
    sender = servicebus_client.get_queue_sender(queue_name=queue_name)
    with sender:     
        sender.send_messages(ServiceBusMessage("done_"+container_name))

def savefile(files, config, pipename, servicebus_client):
    container_name = config['container_name']+pipename
    print(container_name)
    upload(files, config['azure_storage_connectionstring'], container_name)
    with servicebus_client: send_completion_message(servicebus_client, config['queue_name'], container_name)

def get_files(dir):
    #print(os.scandir(dir))
    with os.scandir(dir) as entries:
        for entry in entries:
            if (entry.is_file()) and ((entry.name.endswith(".ply") or entry.name.endswith(".pcd"))):
                #print(entry.name)
                yield entry # turns into a generator function

#######################
# data prep functions #
#######################
#segementation

#Turntable Alg

#ICP

#slice 

#dataprepfinal


"""execute"""
config=load_config(os.path.dirname(os.path.abspath(__file__)))
servicebus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connectionstring'], logging_enable=True)

# #download ply files
#getPLY(config, servicebus_client)

files=get_files(config['save_folder'])
for file in files:
    n, N, file_type = splicename(file.name)
print(n, N, file_type)
# #prepare files

#savefiles to server send message to 


