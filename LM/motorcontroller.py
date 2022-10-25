from azure.servicebus import ServiceBusClient, ServiceBusMessage
import os
import time
import yaml
import numpy as np
# import RPi.GPIO as GPIO
#import GPIO pins

#config
def load_config(dir_config):
    with open(os.path.join(dir_config + r"\config.yaml"), "r") as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)

#azure
def send_completion_message(service_bus_client,queue_name):
    sender = service_bus_client.get_queue_sender(queue_name=queue_name)
    with sender:  
        message="done--turn"   
        sender.send_messages(ServiceBusMessage(message))
        print(f"sent message: {message}")

#turn
def turn(degree,motor_config):
    # GPIO.setmode(GPIO.BCM)
    # GPIO.setup(motor_config["pin_dir"], GPIO.OUT, initial=GPIO.LOW)
    # GPIO.setup(motor_config["pin_pull"], GPIO.OUT, initial=GPIO.LOW)
    steps_required=int(np.ceil(degree*motor_config["steps_per_degree"]))
    for steps in range(steps_required):
        # GPIO.output(motor_config["pin_pull"], GPIO.HIGH)
        # GPIO.output(motor_config["pin_pull"], GPIO.LOW)
        time.sleep(motor_config["delay"])
    print("turned %s degrees with %s steps"%(degree,steps_required))


#setup
current_dir=os.path.dirname(os.path.abspath(__file__))
config=load_config(current_dir)

MAX_WAIT_TIME=config["wait_time"]

pipename_raspberry_pi_to_local= "ralm"
pipename_local_to_raspberry_pi= "lmra"
pipename_local_to_data_prep = "lmdp"
pipename_data_prep_to_local= "dplm"
pipename_data_prep_to_ID_NN= "dpin"
pipename_data_prep_to_ID_reg= "dpir"
pipename_img_gen_to_train= "imtr"
pipename_local_to_img_gen= "lmim"


#wait on message and start turn 

service_bus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connectionstring'])
i=0
while i<MAX_WAIT_TIME:#wait for message
    i+=1
    queue_name =  config['queue_name']+pipename_local_to_raspberry_pi
    receiver = service_bus_client.get_queue_receiver(queue_name=queue_name, max_wait_time=30)
    with receiver:
        messages = receiver.receive_messages(max_wait_time=1)
        if messages:
            message_str=str(messages[0])
            #message=turn--num_of_scan--steps_per_degree--pin_pull--pin_dir--delay
            if message_str.startswith("turn"):
                message_split=message_str.split("--")
                receiver.complete_message(messages[0])
                number_of_scans=int(message_split[1])
                degree=360/number_of_scans
                motor_config={
                    "steps_per_degree":int(message_split[2]),
                    "pin_pull": int(message_split[3]),
                    "pin_dir":int(message_split[4]),
                    "delay": float(message_split[5])
                }

                turn(degree,motor_config)
                queue_name=config["queue_name"]+pipename_raspberry_pi_to_local
                send_completion_message(service_bus_client,queue_name)
                i=0
            else:
                print(f"message kinda wierd ngl -_-: {message_str}")
                receiver.complete_message(messages[0])
        else: 
            print(f"waiting for upload completion for {i} of {MAX_WAIT_TIME} sec.")
            time.sleep(config["sleep_time"])