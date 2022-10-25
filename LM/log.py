
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
from csv import writer

#azure
def load_config(dir_config):
    with open(os.path.join(dir_config + r"\config.yaml"), "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)




current_dir=os.path.dirname(os.path.abspath(__file__))
config=load_config(current_dir)
servicebus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connectionstring'])
log_dir=os.path.join(current_dir,"logs","log2.csv")
receiver = servicebus_client.get_queue_receiver(queue_name="log", max_wait_time=30)
with receiver:
    while True:
        messages = receiver.receive_messages(max_wait_time=1)
        for message in messages:
            with open(log_dir, 'a+', newline='') as write_obj:
                csv_writer = writer(write_obj)
                csv_writer.writerow([message])
                print(message)
            receiver.complete_message(message)
