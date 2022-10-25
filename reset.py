import os
import tqdm
import h5py
from azure.storage.blob import ContainerClient
from azure.core import exceptions
import yaml
from distutils.command.config import config
from tkinter import N
from azure.servicebus import ServiceBusClient, ServiceBusMessage, exceptions 


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
pipename_raspberry_pi_to_local= "ralm"
pipename_local_to_raspberry_pi= "lmra"

current_dir=os.path.dirname(os.path.abspath(__file__))
config=load_config(current_dir)

reset_list=[
    # pipename_local_to_img_gen,
    # pipename_img_gen_to_train_ML,
    # pipename_raspberry_pi_to_local,
    # pipename_local_to_raspberry_pi,
    # pipename_img_gen_to_train_ML,
    pipename_local_to_img_gen,
    # pipename_img_gen_to_train_reg,
]

#reset messaging
servicebus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connectionstring'])

for pipeline_name in reset_list:
    with servicebus_client:
        queue_name=  "queue"+pipeline_name
        receiver = servicebus_client.get_queue_receiver(queue_name=queue_name, max_wait_time=30)
        try:
            with receiver:
                messages = receiver.receive_messages(max_message_count=10000,max_wait_time=30)
                # print(f"messages: {len(messages)}")  
                for message in messages:
                    receiver.complete_message(message)
                    # print(message)
                print(f"cleared {queue_name}")
        except exceptions.ServiceBusAuthenticationError: 
            print(f"queue {queue_name} doesnt exist")
            continue
    #reset containers
    container_name=config["container_name"]+pipeline_name
    container_client = ContainerClient.from_connection_string(config["azure_storage_connectionstring"], container_name)
    if not container_client.exists(): 
        print("wrong connection_string")
    else:
        blobs=container_client.list_blobs()
        for blob in blobs:
            container_client.delete_blob(blob, delete_snapshots="include")
        print(f"cleared {container_name}")