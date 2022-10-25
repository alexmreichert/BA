#imports
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from azure.storage.blob import ContainerClient
from azure.core import exceptions

import os
import time
import yaml
import numpy as np
import open3d as o3d
import json

###############
"""FUNCTIONS"""
###############
#config
def load_config(dir_config):
    with open(os.path.join(dir_config + r"\config.yaml"), "r") as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)

#azure service bus function
def send_turn_message(service_bus_client,queue_name,num_of_scans,motor_config):
    sender = service_bus_client.get_queue_sender(queue_name=queue_name)
    with sender:  
        message="turn--%s--%s--%s--%s--%s"%(str(num_of_scans),str(motor_config["steps_per_degree"]),str(motor_config["pin_pull"]), str(motor_config["pin_dir"]), str(motor_config["delay"]))   #message=turn--num_of_scan--steps_per_degree--pin_pull--pin_dir--delay
        sender.send_messages(ServiceBusMessage(message))
        print(f"sent message: {message}")

def send_upload_message(service_bus_client,queue_name,container_name,degree,config,part,PARTS):
    sender = service_bus_client.get_queue_sender(queue_name=queue_name)
    with sender:  
        #'completion_msg--container_name--float(phi_z)--config/run--int(n)--int(N)'
        message="done--%s--%s--%s--%s--%s"%(container_name,degree,config["config_run"],part,PARTS)
        sender.send_messages(ServiceBusMessage(message))
        print(f"sent message: {message}")

def receive_turn_message(service_bus_client,queue_name,config):
    i=0
    while i<config["wait_time"]:#wait for message
        i+=1
        receiver = service_bus_client.get_queue_receiver(queue_name=queue_name, max_wait_time=30)
        with receiver:
            messages = receiver.receive_messages(max_wait_time=1)
            if messages:
                message_str=str(messages[0])
                #message=turn--num_of_scan--steps_per_degree--pin_pull--pin_dir--delay
                if message_str.startswith("done--turn"): 
                    receiver.complete_message(messages[0])
                    break
                else:
                    print(f"message kinda wierd ngl -_-: {message_str}")
                    receiver.complete_message(messages[0])
                    i=0
            else: 
                print(f"waiting for turn completion for {i} of {MAX_WAIT_TIME} sec.")
                time.sleep(config["sleep_time"])


#azure storage
def upload(pcd, scan_name, connection_string, container_name,path="temp"):
    container_client = ContainerClient.from_connection_string(connection_string, container_name)   
    if not container_client.exists():
        print(f"creating new container with {container_name}")
        try:
            container_client.create_container()
        except exceptions.ResourceExistsError:
            print("waiting for deletion")
            time.sleep(20)
            container_client.create_container()
    print("uploading file:") 
    current_dir=os.path.dirname(os.path.abspath(__file__))
    file_name=os.path.join(current_dir,path,scan_name) 
    o3d.io.write_point_cloud(file_name,pcd)   
    blob_client = container_client.get_blob_client(scan_name)
    with open(file_name, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"{scan_name} uploaded")
    os.remove(file_name)


#scan
def scan():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.json')) as cf:
        rs_config = o3d.t.io.RealSenseSensorConfig(json.load(cf))
    try:
        o3d.t.io.RealSenseSensor.list_devices()
        rs = o3d.t.io.RealSenseSensor()
        rs.init_sensor(rs_config, 0)
        rs.start_capture(True)
        im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
        im_rgbd=im_rgbd.to_legacy() # change to normal open3d format
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        im_rgbd.color, im_rgbd.depth, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])# Flip it, otherwise the pointcloud will be upside down
        rs.stop_capture()
        print("finished scan")
        return pcd
    except RuntimeError:
        print("no cam connected")


#############
"""EXECUTE"""
#############
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

PARTS=48
SCANS=8
MAX_SCANS=SCANS
part_start=40

if config["config_run"]== "config":
    PARTS=1
    SCANS=48
    MAX_SCANS=SCANS/3

motor_config={
                    "steps_per_degree":float(800/360),
                    "pin_pull": 26,
                    "pin_dir":21,
                    "delay": 0.01,
                }

#repeat for N scans wait for space key to start 
#scan (scan, upload, send completion message) and message to raspberry, wait for completion message repeat for N scans

service_bus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connectionstring'])

for part in range(PARTS):
    # part+=part_start
    time_total=time.time()
    input("Press enter to continue...") # wait for user input till going to next scan
    print("scanning part %s"%(part))
    vis = o3d.visualization.Visualizer()
    for scn in range(SCANS):
        if scn==MAX_SCANS-1:break
        pcd = scan() #scan part and return pcd 
        vis.create_window(window_name="scan%sof%s.pcd"%(str(scn+1),MAX_SCANS)) #change the name of the window
        if scn!=0:vis.clear_geometries()
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        send_turn_message(service_bus_client,"queue%s"%(pipename_local_to_raspberry_pi),SCANS,motor_config)# send message to turn
        container_name=config["container_name"]+pipename_local_to_data_prep
        scan_name="config%s_scan%s.pcd"%(part,scn)
        # time.sleep(5)
        upload(pcd,scan_name,config["azure_storage_connectionstring"],container_name)# upload pcd to data prop server or save locally
        send_upload_message(service_bus_client,"queue%s"%(pipename_local_to_data_prep),container_name,float(360/PARTS),config,part,PARTS)#send message to data prep server 
        receive_turn_message(service_bus_client,"queue%s"%(pipename_raspberry_pi_to_local),config)# wait to receive message 
    # send_turn_message(service_bus_client,"queue%s"%(pipename_local_to_raspberry_pi),SCANS,motor_config)# send message to turn
    # receive_turn_message(service_bus_client,"queue%s"%(pipename_raspberry_pi_to_local),config)
    print("total time it took %s"%(time.time()-time_total))
    time.sleep(5)
    vis.destroy_window()