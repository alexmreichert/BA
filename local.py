"""imports"""
from distutils.command.config import config
import queue
from azure.servicebus import ServiceBusClient, ServiceBusMessage
import os
import yaml
import time
from matplotlib import container
from msrestazure import AzureConfiguration
from traitlets import Union
from azure.storage.blob import ContainerClient
from azure.core import exceptions
import pyrealsense2 as rs

"""functions"""
#getply
def get_ply(): 
    # Declare pointcloud object, for calculating pointclouds and texture mappings
    pcd = rs.pointcloud()
    # We want the points object to be persistent so we can display the last cloud when a frame drops
    points = rs.points()

    # Declare RealSense pipeline, encapsulating the actual device and sensors
    pipe = rs.pipeline()
    config = rs.config()
    # Enable depth stream
    config.enable_stream(rs.stream.depth)

    # Start streaming with chosen configuration
    pipe.start(config)

    # We'll use the colorizer to generate texture for our PLY
    # (alternatively, texture can be obtained from color or infrared stream)
    colorizer = rs.colorizer()

    try:
        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()
        colorized = colorizer.process(frames)

        # Create save_to_ply object
        ply = rs.save_to_ply("1.ply")

        # Set options to the desired values
        # In this example we'll generate a textual PLY with normals (mesh is already created by default)
        ply.set_option(rs.save_to_ply.option_ply_binary, False)
        ply.set_option(rs.save_to_ply.option_ply_normals, True)
    	
        print("Saving to 1.ply...")
        # Apply the processing block to the frameset which contains the depth frame and the texture
        ply.process(colorized)
        yield ply
        print("Done")
    finally:
        pipe.stop()

def get_files(dir):
    #print(os.scandir(dir))
    with os.scandir(dir) as entries:
        for entry in entries:
            if (entry.is_file()) and ((entry.name.endswith(".ply") or entry.name.endswith(".pcd"))):
                #print(entry.name)
                yield entry # turns into a generator function

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
    queue_name=config['queue_name']+pipename
    with servicebus_client: send_completion_message(servicebus_client, queue_name, container_name)
    print(f"sent completion message to {queue_name}")

#sendphi
def send_phi(servicebus_client,config, phi):
    queue_name = config['queue_name']
    sender = servicebus_client.get_queue_sender(queue_name=queue_name)
    with sender:     
        sender.send_messages(ServiceBusMessage(str(phi)))

#recieve delta phi plus end
def get_deltaphi():
    deltaphi=0.001
    return deltaphi

#sps
def send_sps():
    print("sps")


"""execute"""
config=load_config(os.path.dirname(os.path.abspath(__file__)))
servicebus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connectionstring'], logging_enable=True)
#ply = get_ply()
ply = get_files(r"C:\Users\alexa\Documents\Uni\Sem8\01_BA\03_Code\TEST\test_pipeline\source")
savefile(ply, config, "lmdp", servicebus_client)


# send_phi()

