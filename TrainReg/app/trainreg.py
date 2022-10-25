#############
#############
"""imports"""
#############
#############

# azure
import yaml
from azure.storage.blob import ContainerClient
from azure.core import exceptions
import re
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from pathlib import Path
from datetime import datetime

# open3d
import time
import open3d as o3d
import os
import numpy as np

#graph stuff
from sklearn.cluster import DBSCAN
from annoy import AnnoyIndex
import pickle

###############
###############
"""functions"""
###############
###############

#io
def load_from (dir):
    pcds=[]
    for file in os.listdir(dir):
            if file.endswith(".pcd"):
                print(f"loading {file}")
                file = os.path.join(dir,file)
                pcd=o3d.io.read_point_cloud(file)
                pcds.append(pcd)
    return pcds

def save_to (pcds, dir, file_name):
    i=0
    for pcd in pcds:
        write_name=os.path.join(dir,f"{file_name[:-4]}_pcd_{i}.pcd")
        print(f"saving {write_name}")
        o3d.io.write_point_cloud(write_name, pcd)
        i+=1

def get_files(dir):
    with os.scandir(dir) as entries:
        for entry in entries:
            if entry.is_file():
                yield entry # turns into a generator function

#azure
def download(connection_string, container_name, save_path, download_one=True):
    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    labels=[]
    if not container_client.exists(): 
        print("wrong connection_string")
        return None
    else:
        blobs=container_client.list_blobs()
        for blob in blobs:
            filename=os.path.join(save_path,blob.name)
            if Path(filename).exists():
                labels.append(filename)
            else:
                blob_name=blob.name
                StorageStreamDownloader = container_client.download_blob(blob)
                #print(StorageStreamDownloader)
                try:
                    file = open(filename, 'wb')
                except FileNotFoundError:
                    os.mkdir(save_path)
                    file = open(filename, 'wb')
                data=StorageStreamDownloader.readall()
                print(f"saving localy and deleting blob: {blob.name}")
                # container_client.delete_blob(blob, delete_snapshots="include")
                file.write(data)
                file.close()
                labels.append(filename)
                if download_one: break
    return labels

def get_pcd(config, servicebus_client, pipeline_name, MAX_WAIT_TIME, skip_message=True):
    changestr="None"
    config_bool=False
    labels=[]
    recieved=False
    #wait for completion message and download file
    if not skip_message:
        with servicebus_client:
            # get the Subscription Receiver object for the subscription 
            queue_name=  config['queue_name']+pipeline_name
            receiver = servicebus_client.get_queue_receiver(queue_name=queue_name, max_wait_time=30)
            with receiver:
                messages = receiver.receive_messages(max_wait_time=1)
                i=1
                while ((not messages) or str(messages[0])[:4]!="done") and i<MAX_WAIT_TIME: 
                    print(f"waiting for upload completion for {i} of {MAX_WAIT_TIME} sec.")
                    time.sleep(1)
                    messages = receiver.receive_messages(max_wait_time=1)
                    i+=1
                if messages:
                    for message in messages:
                        strmsg=str(message)
                        #'done--{container_name}--{label}--{config}--{changestr}'
                        #done--containerlmim--Bauteil-013.stl--True--2.2_10
                        matches = re.search(r'(\w+)\-\-(\w+)\-\-(\w+\-\d+)\.\w+\-\-(\w+)\-\-(\S+)',strmsg)
                        completion_msg=matches.group(1)
                        container_name=matches.group(2)
                        label=matches.group(3)
                        change_config=matches.group(4)
                        config_bool=    change_config=="True"
                        if config_bool: changestr=matches.group(5)
                        if(completion_msg=="done"):
                            recieved=True
                            print(f"starting download from container: {container_name}, label: {label}")
                            download(config['azure_storage_connectionstring'],container_name, os.path.join(os.path.dirname(os.path.abspath(__file__)),config['save_folder']))
                            print(f"finished download from container: {container_name}")
                            labels.append(label)
                            receiver.complete_message(message)
                            continue
                        else:
                            print(f"message kinda wierd ngl -_-: {strmsg}")
                            receiver.complete_message(message)
                            continue
                else: print("no message next run")
    else:
        container_name=config["container_name"]+pipeline_name
        labels=download(config['azure_storage_connectionstring'],container_name, os.path.join(os.path.dirname(os.path.abspath(__file__)),config['save_folder']), download_one=False)
    return labels, recieved, config_bool, changestr

def load_config(dir_config):
    with open(os.path.join(dir_config, "config.yaml"), "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)

def send_completion_message(servicebus_client,queue_name, container_name,label,change_config="False", configstr="None"):
    sender = servicebus_client.get_queue_sender(queue_name=queue_name)
    with sender:  
          #'done--{container_name}--{Bauteil-19.stl}--{True}--{changestring}'
          #changestring=r_n_s_v_slicevariation_mode         else none
        message=f"done--{container_name}--{label}--{change_config}--{configstr}"   
        sender.send_messages(ServiceBusMessage(message))
        print(f"sent message: {message}")
     
def edit_config(changestr):
    # 1_0_0_0_0_6000_reg
    matches = re.search(r'(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\w+)',changestr)
    r=matches.group(1)
    n=matches.group(2)
    d=matches.group(3)
    b=matches.group(4)
    slice_variations=matches.group(5)
    points=matches.group(6)
    mode=matches.group(7)

    print(f"editing config file to r: {r}, n: {n}, d: {d}, b: {b}, slice_variations:{slice_variations}, mode: {mode}")
    with open("config.yaml") as f:
        list_doc = yaml.safe_load(f)
    # print(f"list_doc: {list_doc}")
    
    list_doc["r"] = int(r)
    list_doc["n"] = int(n)
    list_doc["d"] = int(d)
    list_doc["b"] = int(b)
    list_doc["slice_variations"] = int(slice_variations)
    list_doc["points"] = int(points)
    list_doc["mode"] = mode

    with open("config.yaml", "w") as f:
        yaml.dump(list_doc, f)

def upload(files,upload_folder, connection_string, container_name,queue_name, servicebus_client, first_message_different=False,configstr="None"):

    container_client = ContainerClient.from_connection_string(connection_string, container_name)   
    if not container_client.exists():
        print(f"creating new containter with {container_name}")
        try:
            container_client.create_container()
        except exceptions.ResourceExistsError:
            print("waiting for deletion")
            time.sleep(20)
            container_client.create_container()
    i=0
    print("uploading files:")     
    for file in files:
        file_path=os.path.join(upload_folder,file)
        blob_client = container_client.get_blob_client(file)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
            print(f"{file} uploaded")
            # if i==0 and first_message_different:send_completion_message(servicebus_client,queue_name, container_name,file.name,change_config=first_message_different,configstr=configstr)
            # else: send_completion_message(servicebus_client,queue_name, container_name,file.name)
            i+=1
        try:
            os.remove(file_path)
            print(f"deleted {file}")
        except PermissionError:
            print(f"could not delete {file} due to lacking Permission")
            continue

def log(service_bus_client,log,sender_info="TRAINREG-"):
    sender = service_bus_client.get_queue_sender(queue_name="log")
    message=sender_info+str(datetime.now())+": "+log
    with sender:  
        sender.send_messages(ServiceBusMessage(message))
        print("log: %s"%(message))

#creategraph
def create_graph(model,data): # dbscan model and points 
    #initialize 
    graph_feature={}
    cluster_points={}

    #create get all points in a cluster then create dictionary for the averages of each cluster
    for label in np.unique(model.labels_):
        points_in_cluster=data[np.nonzero(model.labels_==label)]
        cluster_points[label]=np.mean(points_in_cluster, axis=0) # averages all points
    
    #calculate the distance between the all of the clusters 
    for label in np.unique(model.labels_):
        node_point=cluster_points[label] # get average from cluster
        labels=list(cluster_points.keys()) # get list of all cluster labels
        points_nodes=np.asarray(list(cluster_points.values())) # get list of all cluster value
        vector=np.asarray(points_nodes)-np.ones((len(points_nodes),1))*node_point  #find vector difference
        distance_edges=np.sqrt(np.sum(np.square(vector),axis=1)) # euclidiean difference 
        edges=np.c_[distance_edges,labels,] #create numpy array with labels connected with euclidian distance
        edges=edges[edges[:,1]!=label] # remove distance zeros
        edges=edges[edges[:, 0].argsort()] # sort distance
        graph_feature[label]=edges # save to dictionary 
    return graph_feature


#############
#############
"""execute"""
#############
#############
pipename_local_to_data_prep = "lmdp"
pipename_data_prep_to_local= "dplm"
pipename_data_prep_to_ID_NN= "dpin"
pipename_data_prep_to_ID_reg= "dpir"
pipename_img_gen_to_train_reg= "imtr"
pipename_img_gen_to_train_ML= "imtm"
pipename_local_to_img_gen= "lmim"
pipename_train_reg_to_id_reg="trir"
pipename_train_ML_to_id_ML="tmim"

#test variables
use_local_files=False
print_stuff=False

#load yaml config file
config=load_config(os.path.dirname(os.path.abspath(__file__)))
upload_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)),config["upload_folder"])

#azure servicebus messaging client
servicebus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connectionstring'])


variation_dict={
    "points":[1024,4096],
    "radius": [0.01,0.02],
    "feature": [0.002,0.002],
    "noise": [0.0001,0.0004],
}

#mainloop
total_time=time.time()
max_runs=config['max_runs']
run=0
labels=[]
load_files=path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"download")
while run<max_runs:
    print(f"-----------------------Running {max_runs-run} runs more---------------------------------")
    t_run=time.time()
    if not use_local_files:
        #get message download pcd
        labels, recieved_mes, config_bool, changestr=get_pcd(config, servicebus_client, pipename_img_gen_to_train_reg, config['wait_time'])
        #add pcd 
        if config_bool: 
            edit_config(changestr,os.path.dirname(os.path.abspath(__file__)))
            config=load_config(os.path.dirname(os.path.abspath(__file__)))
    run+=1
    if len(labels)!=0:break

#seperate into variants
variants=[]
file_dict={}
radius_dict={}
for label in labels:
    part_name=label.split("/")[-1].split(".")[0]
    label_splits=part_name.split("_")
    variant=""
    radius_str=""
    for label_split in label_splits:
        if label_split.startswith("variation"):variant=label_split
        if label_split.startswith("radius"):radius_str=label_split
    if variant in file_dict:
        file_dict[variant]=file_dict[variant]+[(part_name,label)]
    else:
        file_dict[variant]=[(part_name,label)]
    radius_dict[variant]=int(radius_str.split("-")[1])

#loop through variants
for variant in list(file_dict.keys()):
    log(servicebus_client, variant)      
    #get radius
    radius_feature=variation_dict["feature"][radius_dict[variant]]   
    print(radius_feature)
    #load pcd from folder into memory
    pcds=[]
    filenames=[]
    for part, file in file_dict[variant]:        
        pcd = o3d.io.read_point_cloud(file)
        pcd_down = pcd.voxel_down_sample(radius_feature)#voxel sampling
        pcds.append(pcd_down)
        filenames.append(part)

    # log(servicebus_client, "filenames: %s"%(str(filenames)))

    #get features and annoyindex 
    data=[] # [pcd, feature stuff, ANNOY graph]
    for index_file in range(len(pcds)):
        pcd=pcds[index_file]

        
        file_name=filenames[index_file].split("/")[-1]
        print(file_name)
        pcd_points=np.asarray(pcd.points)
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)# higher radius = more feature found with mroe points
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param) #returns open3d.pipelines.registration.Feature
        feature_points=pcd_fpfh.data.T # feature.data returns dim x n float64 numpy array dim = number of features (max 33) and n = number of points 
        feature_pcd={} # feature number: graph, model.labels, feature points
        #loop through each feature found by FPFH, cluster points, create graoh
        
        for feature in range(feature_points.shape[1]):
            # log(servicebus_client, f"feature: {feature}")
            #get index where feature exists            
            index_pcd_feature=np.nonzero(feature_points[:,feature] !=0)
            print(index_pcd_feature)
            #grab points to those indexes
            points_pcd_feature=pcd_points[index_pcd_feature]
            graph_feature_pcd={}
            model = DBSCAN(eps=radius_feature, min_samples=1)
            #check if features have been found 
            empty=True
            if len(points_pcd_feature)==0: 
                print("no feature found")
                # log(servicebus_client, "no features found")
            else:
                empty=False
                feature_pcd_select=o3d.geometry.PointCloud()
                feature_pcd_select.points=o3d.utility.Vector3dVector(points_pcd_feature)
                feature_pcd_select.paint_uniform_color([0,1,0])
                o3d.visualization.draw_geometries([feature_pcd_select,pcd])
                model = DBSCAN(eps=radius_feature, min_samples=1)
                model.fit_predict(points_pcd_feature) # dbscan model .labels_ returns all labels
                dbscan_labels=model.labels_
                print(f"found {len(np.unique(model.labels_))} clusters pcd")
                graph_feature_pcd=create_graph(model,points_pcd_feature)  
            if empty:dbscan_labels=[]
            feature_pcd[feature]=[graph_feature_pcd,dbscan_labels,points_pcd_feature]
        #create annoy graph
        t = AnnoyIndex(3, 'euclidean')

        for i in range(len(pcd_points)):
            v = pcd_points[i,:]
            t.add_item(i, v)
        t.build(10)
        t.save(os.path.join(upload_folder,f'{file_name}annoy.ann'))
        o3d.io.write_point_cloud(os.path.join(upload_folder,f'{file_name}.pcd'),pcd)
        #save data to list
        data.append([f"{file_name}.pcd",feature_pcd, f"{file_name}annoy.ann"]) #files, 
    timetosave=time.time()

    
    #pickle list to upload
    part, file in file_dict[variant]
    data_name="_".join((part.split("/")[-1]).split("_")[:-3])+"_data_.pkl"
    print(os.path.join(upload_folder,data_name))
    with open(os.path.join(upload_folder,data_name), 'wb') as f:
        pickle.dump(data, f)
    log(servicebus_client, f"took {time.time()-timetosave} seconds to save pickle")

    #upload pickle
    files=os.listdir(upload_folder)
    log(servicebus_client, "files: %s"%(str(files)))
    containername=config["container_name"]+pipename_train_reg_to_id_reg
    queuename=config["queue_name"]+pipename_train_reg_to_id_reg
    upload(files,upload_folder, config["azure_storage_connectionstring"],containername,queuename,servicebus_client)      
    log(servicebus_client, f"-----------------------Run took: {time.time()-t_run} seconds---------------------------------")
log(servicebus_client, f"Time it took in total: {time.time()-total_time}")

