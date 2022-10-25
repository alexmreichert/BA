#############
#############
"""imports"""
#############
#############

# azure
from pathlib import Path
import yaml
from azure.storage.blob import ContainerClient
from azure.core import exceptions
import re
from azure.servicebus import ServiceBusClient, ServiceBusMessage
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

#TEASER
import copy
import pandas as pd
from zmq import QUEUE 
import teaserpp_python #linux only


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
def download_data(connection_string, container_name, save_path,variation, labels):
    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    files=[]
    if not container_client.exists(): 
        print("wrong connection_string")
        return None
    else:
        blobs=container_client.list_blobs()
        for blob in blobs: # get all blobs in container
            blob_name=blob.name

            #parse blob_name
            if not ("variation-%s_"%(variation) in blob_name): continue
            
            download_blob=False
            if blob_name.endswith("pkl"): download_blob = True
            elif blob_name.endswith("pcd") or blob_name.endswith("ann"):
                for blob_name_split in blob_name.split("_"):
                    if blob_name_split.startswith("Bauteil"):
                        if int(blob_name_split.split("-")[1]) in labels:
                            download_blob=True
                            break
            if download_blob:
                filename=os.path.join(save_path,blob_name)
                if Path(filename).exists():
                    files.append(blob_name)
                else:
                    StorageStreamDownloader = container_client.download_blob(blob)
                    try:
                        file = open(filename, 'wb')
                    except FileNotFoundError:
                        os.mkdir(save_path)
                        file = open(filename, 'wb')
                    data=StorageStreamDownloader.readall()
                    print(f"saving locally: {blob_name}")
                    file.write(data)
                    file.close()
                    files.append(blob_name)
        del blobs
    return files

def download_part(connection_string, container_name, save_path,name_part=None):
    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    if not container_client.exists(): 
        print("wrong connection_string")
        return None
    else:
        blobs=container_client.list_blobs()
        for blob in blobs:
            blob_name=blob.name
            if (name_part is not None) and (blob_name!=name_part):continue
            filename=os.path.join(save_path,blob_name)
            if not Path(filename).exists():
                StorageStreamDownloader = container_client.download_blob(blob)
                try:
                    file = open(filename, 'wb')
                except FileNotFoundError:
                    os.mkdir(save_path)
                    file = open(filename, 'wb')
                data=StorageStreamDownloader.readall()
                print(f"saving locally: {blob_name}")
                file.write(data)
                file.close()
        del blobs

def load_config(dir_config):
    with open(os.path.join(dir_config, "config.yaml"), "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)

def send_message(servicebus_client,queue_name, message):
    sender = servicebus_client.get_queue_sender(queue_name=queue_name)
    with sender:  
        sender.send_messages(ServiceBusMessage(message))
        log(servicebus_client,f"sent message: {message}")
     
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

def upload(files, connection_string, container_name,queue_name=None, servicebus_client=None, first_message_different=False,configstr="None"):

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
        blob_client = container_client.get_blob_client(file.name)
        with open(file.path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
            print(f"{file.name} uploaded")
            i+=1
        os.remove(file.path)

def log(service_bus_client,log,sender_info="ID REG-"):
    sender = service_bus_client.get_queue_sender(queue_name="log")
    message=sender_info+" "+str(datetime.now())+": "+log
    with sender:  
        sender.send_messages(ServiceBusMessage(message))
        print("log: %s"%(message))

#get features, graph
def get_features(pcd,config,feature_radius):
    pcd_points=np.asarray(pcd.points)
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100)# higher radius = more feature found with mroe points
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param) #returns open3d.pipelines.registration.Feature
    feature_points=pcd_fpfh.data.T # feature.data returns dim x n float64 numpy array dim = number of features (max 33) and n = number of points 
    feature_pcd={} # feature number: graph, model.labels, feature points
    #loop through each feature found by FPFH, cluster points, create graph
    for feature in range(feature_points.shape[1]):
        if print_stuff:print(f"feature: {feature}")
        #get index where feature exists            
        index_pcd_feature=np.nonzero(feature_points[:,feature] !=0)
        #grab points to those indexes
        points_pcd_feature=pcd_points[index_pcd_feature]
        graph_feature_pcd={}
        model = DBSCAN(eps=feature_radius, min_samples=1)
        #check if features have been found 
        empty=True
        if len(points_pcd_feature)==0: 
            if print_stuff: print("no features found")
        else:
            empty=False
            model.fit_predict(points_pcd_feature) # dbscan model .labels_ returns all labels
            dabscan_labels=model.labels_
            if print_stuff: print(f"found {len(np.unique(model.labels_))} clusters pcd1")
            graph_feature_pcd=create_graph(model,points_pcd_feature)  
        if empty:dabscan_labels=[]
        feature_pcd[feature]=[graph_feature_pcd,dabscan_labels,points_pcd_feature]
    return feature_pcd

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

#get correspondance pairs
def get_correspondance_pair(pairs, labels1, labels2, data1, data2):
    correspondance_points=[]
    correspondance_lines=[]
    for node_i in range(len(pairs)):
        node=pairs[node_i]        
        index1=np.asarray(np.nonzero(labels1==node[0])).flatten()
        index2=np.asarray(np.nonzero(labels2==node[1])).flatten()
        prune_length=min(len(index1),len(index2))
        index1=index1[:prune_length]
        index2=index2[:prune_length]
        for i in range(prune_length):
            correspondance_points.append(data1[index1[i]])
            correspondance_points.append(data2[index2[i]])
            length=len(correspondance_lines)
            correspondance_lines.append([length*2,length*2+1])
    return correspondance_points,correspondance_lines

#score function
def get_score(annoyindex,points, print_stuff=False, vizualize_stuff=False, og_points=np.zeros((1,3))):
    NN_lines=np.zeros((len(points),2))
    NN_distances=np.zeros((len(points)))
    t_var=time.time()
    for NN_index in range(len(points)):
        index, dst = annoyindex.get_nns_by_vector(points[NN_index,:], 1, include_distances=True)
        if vizualize_stuff:NN_lines[NN_index,:]    =   [len(og_points)+NN_index,index[0]]
        NN_distances[NN_index]  =   dst[0]
    if print_stuff: print(f"took {time.time()-t_var} seconds to find nearest neighbors")
    score = np.sum(NN_distances/len(points))
    if vizualize_stuff:
        points=np.concatenate((og_points,points))
        colors = [[0,0,1] for i in NN_lines]
        lineset=o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(NN_lines),
        )
        lineset.colors = o3d.utility.Vector3dVector(colors)
        return score, lineset
    return score

def confidence(scores,labels):
    total_score=0
    label=labels[np.argmin(scores,axis=0)]
    sum=np.sum(np.asarray(scores))
    confidence=(sum-np.asarray(scores).min(axis=0))/sum
    return label,confidence

#init function
def init(servicebus_client, config):
    #wait for message
    queue_name=  config['queue_name']+"dpir"
    received_init_message=False
    while not received_init_message:
        servicebus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connectionstring'])
        receiver = servicebus_client.get_queue_receiver(queue_name=queue_name, max_wait_time=30)
        with receiver:
            messages = receiver.receive_messages(max_wait_time=1)
            for message in messages:
                if str(message).startswith("init"): # pars message
                    message_split=str(message).split("_")
                    container_name=int(message_split[1].split("-")[1])
                    labels=[]
                    for label_str in message_split[2].split("-")[1:]:
                        labels.append(int(label_str))
                    variation=int(message_split[3].split("-")[1])
                    received_init_message=True
                    try:
                        receiver.complete_message(message)
                    except Exception as e: 
                        log(servicebus_client, str(e), "ID REG - %s"%(CONTAINER_NAME))

                    return container_name,labels,variation
        time.sleep(1)
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
pipename_id_reg_to_local="irlm"

#for factorial experiment
variation_dict={
    "points":[1024,4096],
    "radius": [0.01,0.02],
    "feature": [0.001,0.002],
    "noise": [0.05,0.15],
}

#container specific state
CONTAINER_NAME=None #int
LABELS=None         #list of int
VARIATION=None      #int
QUEUE_NAME=""       #string

#load yaml config file
config=load_config(os.path.dirname(os.path.abspath(__file__)))

#local oathas
path_upload=os.path.join(os.path.dirname(os.path.abspath(__file__)),config["upload_folder"])
path_save=os.path.join(os.path.dirname(os.path.abspath(__file__)),config["save_folder"])  
path_scan=os.path.join(os.path.dirname(os.path.abspath(__file__)),config["save_folder_scan"])

#test variables
print_stuff=config["print_stuff"]
vizualize_stuff=config["vizualize_stuff"]

#azure servicebus messaging client
servicebus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connectionstring'])

#mainloop
while True:
    total_time=time.time()

    #########
    #messages
    #########
    if CONTAINER_NAME is None or QUEUE_NAME is None or VARIATION is None:#check if container name exists
        log(servicebus_client, "initializing container", "ID REG - X")
        CONTAINER_NAME,LABELS, VARIATION=init(servicebus_client, config)#initialize container        
        QUEUE_NAME=config["queue_name"]+pipename_data_prep_to_ID_reg+str(CONTAINER_NAME) #queue name with container number
        log(servicebus_client, "container %s initialized with labels: %s and variation: %s"%(CONTAINER_NAME, str(LABELS), VARIATION), "ID REG - %s"%(CONTAINER_NAME))

    #get part_name and 
    id_message_received=False
    part_name=""
    storage_container_name_dp=""
    log(servicebus_client, "Getting Id message", "ID REG - %s"%(CONTAINER_NAME))
    while not id_message_received:#wait for ID request and download point cloud, if init message change variation
        print("getting part message")
        servicebus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connectionstring'])
        receiver = servicebus_client.get_queue_receiver(queue_name=QUEUE_NAME, max_wait_time=30)
        with receiver:
            messages = receiver.receive_messages(max_wait_time=1)
            for message in messages:
                print(str(message))
                if str(message).startswith("init"): # pars init message
                    receiver.complete_message(message)
                    message_split=str(message).split("_")
                    CONTAINER_NAME=int(message_split[1].split("-")[1])
                    LABELS=[]
                    for label_str in message_split[2].split("-")[1:]:
                        LABELS.append(int(label_str))
                    VARIATION=int(message_split[3].split("-")[1])
                    id_message_received=False
                    log(servicebus_client, "container %s initialized with labels: %s and variation: %s"%(CONTAINER_NAME, str(LABELS), VARIATION), "ID REG - %s"%(CONTAINER_NAME))

                elif str(message).startswith("id"): # pars id message
                    try:
                        receiver.complete_message(message)
                    except Exception as e: 
                        log(servicebus_client, str(e), "ID REG - %s"%(CONTAINER_NAME))
                    message_split=str(message).split("_")
                    part_name=message_split[1]
                    storage_container_name_dp=message_split[2]
                    id_message_received=True
                    log(servicebus_client, "recieved message: %s"%(str(message)), "ID REG - %s"%(CONTAINER_NAME))

                else: 
                    receiver.complete_message(message)
                    print("message looks wierd")
                    log(servicebus_client, "recieved weird message: %s"%(str(message)), "ID REG - %s"%(CONTAINER_NAME))
        time.sleep(1)
    #########
    #download
    #########

    #download part
    download_part(config['azure_storage_connectionstring'],storage_container_name_dp, path_scan,name_part=part_name)                   
    log(servicebus_client, "Downloaded part %s"%(part_name), "ID REG - %s"%(CONTAINER_NAME))

    #download point clouds, annoy trees and pickles unless already there
    storage_container_name_tr=config["container_name"]+pipename_train_reg_to_id_reg
    files_data=download_data(config['azure_storage_connectionstring'], storage_container_name_tr, path_save,VARIATION, LABELS)
    log(servicebus_client, "Downloaded %s data files"%(len(files_data)), "ID REG - %s"%(CONTAINER_NAME))
    

    #get pkl name
    pkl_file_name=""
    list_of_parts=[]
    for file in os.listdir(path_save):
        file_split=file.split(".")
        if "variation-%s"%(str(VARIATION)) in file_split[0].split("_") and file.endswith(".pkl"):
            pkl_file_name=file
        for property in file_split[0].split("_"):   
            if (property.split("-")[0]=="Bauteil") and not (property in list_of_parts):
                list_of_parts.append(property)

    print("pkl_file_name: %s"%(pkl_file_name))
    log(servicebus_client,"list_of_parts: %s"%(list_of_parts), "ID REG - %s"%(CONTAINER_NAME))

    result={}
    labels_part=[]#labels

    noise_bound=0.05
    feature_radius=0.001
    voxel_down_sample=True
    #parse noise and radius
    for pkl_split in pkl_file_name.split("_"):
        split_prop=pkl_split.split("-")
        if split_prop[0]=="noise":noise_bound=variation_dict["noise"][int(split_prop[1])]
        if split_prop[0]=="feature":voxel_down_sample=0==int(split_prop[1])
    log(servicebus_client,"voxel: %s"%(str(voxel_down_sample)), "ID REG - %s"%(CONTAINER_NAME))
    #Teaser++ setup 
    # Populate the parameters
    solver_params = teaserpp_python   .RobustRegistrationSolver.Params()
    solver_params.cbar2 = config["cbar2"]
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver   .ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12


    #load data from folder into memory
    data=[]
    time_to_load=time.time()
    
    #unpack pickle 
    print("pkl_file_name: %s"%(pkl_file_name))
    with open(os.path.join(path_save,pkl_file_name), 'rb') as f:
        data = pickle.load(f)
    print(f"took {time.time()-time_to_load} seconds to load pickle")
    print("data has %s row"%(len(data)))

    #load pcd to identify into memory from pcd dic # should only be one
    
    pcd_scan=o3d.io.read_point_cloud(os.path.join(path_scan,part_name))
    if voxel_down_sample:pcd_scan =pcd_scan.voxel_down_sample(feature_radius)
    pcd_scan.estimate_normals()
    pcd_scan.translate([100,100,100])

    #get features and create graph
    feature_pcd_scan= get_features(pcd_scan, config,feature_radius)  #dictionary - feature number: graph, model.labels, feature points
    #pcd to compare tp 
    row_number=0
    for row in data: # part by part #[pcd files, features, tree files ]
        label_row=row[0][:-4].split("_")[-3] # parse Bauteil name

        if not label_row in list_of_parts: continue#check if in list_of_parts else skip

        if not (label_row in labels_part): # add file_name to new list
            labels_part.append(label_row)
        
        time_per_pcd=time.time() 
        pcd=o3d.io.read_point_cloud(os.path.join(path_save,row[0])) 
        feature_pcd=row[1]
        print(f"feature_pcd has %s row"%(len(feature_pcd)))

        #load annoy tree
        annoy_index = AnnoyIndex(3, 'euclidean')
        annoy_index.load(os.path.join(path_save,row[2])) # super fast, will just map the file
        points_features=[]
        line_features=[]

        #loop through each feature found in the scan and compare it with the pcd
        for key in feature_pcd_scan.keys():
            #unpack the feature_pcd list
            graph_scan, labels_scan, feature_points_scan=feature_pcd_scan[key] # unpack feature newly generated
            graph, labels, feature_points=feature_pcd[key] # unpack feature downloaded
            #check if feature is empty
            if len(feature_points)==0 or len(feature_points_scan)==0: 
                if print_stuff:print(f"no feature {key} found")
                del graph_scan, labels_scan, feature_points_scan, graph, labels, feature_points
            else: 
                prune_len=min(len(graph.keys()),len(graph_scan.keys()))-1 #maximum correspondance pairs
                
                
                difference_nodes=[] 
                t=time.time()
                #compare nodes in the graph
                count_comparison=0
                for key1 in graph.keys():   # loop thru all cluster labels in feature 1
                    M1_node =graph[key1]   #Get the node with all distances 
                    for key2 in graph_scan.keys(): # loop thru all cluster labels in feature 2 
                        M2_node =graph_scan[key2] #node from second graph
                        difference=abs(np.sum(M2_node[:prune_len,0]-M1_node[:prune_len,0])) #difference 
                        difference_nodes.append([np.c_[np.r_[M1_node[:prune_len,1],key1],np.r_[M2_node[:prune_len,1],key2]],difference]) #key1 
                        count_comparison+=1
                        del M2_node, difference
                        if count_comparison>10000:break
                    if count_comparison>10000:break #stop ram overflow
                    del M1_node
                del graph_scan, graph
                difference_nodes_np=np.asarray(difference_nodes)#convert to numpy
                #node with smallest sum ist selected
                del difference_nodes
                
                best_node_match=difference_nodes_np[np.argmin(difference_nodes_np[:,1]),0]
                del difference_nodes_np
                # log(servicebus_client ,f"finding best node match took {time.time()-t} seconds, found match with {len(best_node_match)} matches, took {count_comparison} counts")
                t_pairs=time.time()
                #based on matches points of the correspondance pairs are found
                points, _ = get_correspondance_pair(best_node_match, labels, labels_scan, feature_points, feature_points_scan)
                points_features+=points
                if points: del points# log(servicebus_client ,f"getting pairs took {time.time()-t_pairs} seconds")
                del labels_scan, feature_points_scan, labels, feature_points
                del best_node_match
                
        log(servicebus_client ,f"Time it took to match pcd: {time.time()-time_per_pcd} with {len(points_features)} points", "ID REG - %s"%(CONTAINER_NAME)) 
        #TEASER Registration
        #move source (scan) to target (generated )
        points_features_np=np.asarray(points_features[::2]) # every second point
        points_features_np_scan=np.asarray(points_features[1::2])
        print("src shape: %s"%str(points_features_np.shape))
        
        if len(points_features_np.shape)!=2: # if no features are found random sample points as correspondance pairs
            points_features_np      =np.asarray(pcd.points)
            points_features_np_scan =np.asarray(pcd_scan.points)
            print(len(points_features_np_scan))
            prune_length=min(len(points_features_np),len(points_features_np_scan))
            print(prune_length)
            points_features_np=points_features_np[:prune_length]
            points_features_np_scan=points_features_np_scan[:prune_length]
            del prune_len
        
        tar=points_features_np.T
        src=points_features_np_scan.T
        del points_features_np
        del points_features_np_scan

        #randomly sample correspondance to limit number of pairs
        print(src.shape[1])
        print(tar.shape[1])
        index_src=np.arange(src.shape[1])
        np.random.shuffle(index_src)
        src=src[:,index_src[:1000]]
        tar=tar[:,index_src[:1000]]
        del index_src
        print(src.shape[1])
        print(tar.shape[1])

        #teaser
        t_teaser=time.time()
        solver_rotation = teaserpp_python.RobustRegistrationSolver(solver_params)
        solver_rotation.solve(src, tar)
        solution_rotation = solver_rotation.getSolution() 

        pcd_src=o3d.geometry.PointCloud()
        pcd_src.points=o3d.utility.Vector3dVector(src.T)
        pcd_src.rotate(solution_rotation.rotation)
        src_translation=np.asarray(pcd_src.points).T
        del pcd_src
        solver_translation = teaserpp_python.RobustRegistrationSolver(solver_params)
        solver_translation.solve(src_translation, tar)
        solution_translation = solver_translation.getSolution()   
        del src_translation

        pcd_scan_registration = copy.deepcopy(pcd_scan)
        pcd_scan_registration.rotate(solution_rotation.rotation)
        pcd_scan_registration.translate(solution_translation.translation)
        log(servicebus_client, "it took %s seconds to run teaser twice"%(time.time()-t_teaser), "ID REG - %s"%(CONTAINER_NAME))
        
        
        if False: # visualization and testing stuff
            #vizualize correspondance pairs by saving them as csv files
            save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),f"upload_test")
            # #check correspondance pairs 
            # max_clique = solver_rotation.getTranslationInliersMap()
            # inlierpoints=np.zeros([2*src.T.shape[0],3])
            # ogpoints=np.zeros([2*src.T.shape[0],3])
            # oglines=np.zeros([src.T.shape[0],2])
            # outlierpoints=np.zeros([2*src.T.shape[0],3])
            # inlierlines=np.zeros([src.T.shape[0],2])
            # outlierlines=np.zeros([src.T.shape[0],2])
            # print(f"max_clique: {max_clique}")

            # for i in range(tar.T.shape[0]):
            #     if i in max_clique:
            #         inlierpoints[2*i,:]=src.T[i,:]
            #         inlierpoints[2*i+1,:]=tar.T[i,:]
            #         inlierlines[i,:]=[2*i,2*i+1]
            #     else: 
            #         outlierpoints[2*i,:]    =src.T[i,:]
            #         outlierpoints[2*i+1,:]  =tar.T[i,:]
            #         outlierlines[i,:]       =[2*i,2*i+1]
            #     oglines[i,:]   =[2*i,2*i+1]

            
            # pd.DataFrame(src.T).to_csv(os.path.join(save_path,f"csv{row_number}src.csv"))
            # pd.DataFrame(tar.T).to_csv(os.path.join(save_path,f"csv{row_number}tar.csv"))

            # pd.DataFrame(inlierpoints).to_csv(os.path.join(save_path,f"csv{row_number}inlierpoints.csv"))
            # pd.DataFrame(inlierlines).to_csv(os.path.join(save_path,f"csv{row_number}inlierlines.csv"))
            # pd.DataFrame(outlierpoints).to_csv(os.path.join(save_path,f"csv{row_number}outlierpoints.csv"))
            # pd.DataFrame(outlierlines).to_csv(os.path.join(save_path,f"csv{row_number}outlierlines.csv"))

            # pd.DataFrame(np.asarray(pcd.points)).to_csv(os.path.join(save_path,f"csv{row_number}points1.csv"))
            # pd.DataFrame(np.asarray(pcd_scan.points)).to_csv(os.path.join(save_path,f"csv{row_number}points2.csv"))

            # # Print the solution
            print(f"Rotation: {solution_rotation.rotation}")
            print(f"Translation: {solution_translation.translation}")


            pcd.paint_uniform_color([0,0,1])
            pcd_scan.paint_uniform_color([1,0,0])
            pcd2 = copy.deepcopy(pcd_scan)
            pcd2.rotate(solution_rotation.rotation)
            pcd2.paint_uniform_color([0,1,1])
            # o3d.io.write_point_cloud(os.path.join(save_path,f"{row_number}ogpcd3.pcd"),pcd2)
            # pcd3 = copy.deepcopy(pcd2)
            # pcd3.translate(solution_translation.translation)
            # pcd3.paint_uniform_color([0,1,0])
            o3d.io.write_point_cloud(os.path.join(save_path,f"{row_number}ogpcdtarget.pcd"),pcd)
            #o3d.io.write_line_set(os.path.join("final","oglineset.ply"),correspondance_linesets)
            o3d.io.write_point_cloud(os.path.join(save_path,f"{row_number}ogpcdsource.pcd"),pcd_scan)
            o3d.io.write_point_cloud(os.path.join(save_path,f"{row_number}ogpcdrotation.pcd"),pcd2)
            o3d.io.write_point_cloud(os.path.join(save_path,f"{row_number}ogpcdtranslation.pcd"),pcd_scan_registration)
            row_number+=1
            
            container_name_test=config["container_name"]+"test"
            # container_client_test = ContainerClient.from_connection_string(config["azure_storage_connectionstring"], container_name_id)   
            files_test=get_files(save_path)
            upload(files_test,config["azure_storage_connectionstring"],container_name_test)
        
        #remove stuff from memorry
        score=get_score(annoy_index,np.asarray(pcd_scan_registration.points),print_stuff=print_stuff)
        del src,tar
        del solver_translation
        del pcd_scan_registration
        log(servicebus_client ,"the score for %s is %s"%(label_row,score), "ID REG - %s"%(CONTAINER_NAME))
        result[label_row]=score
        del score
    del feature_pcd_scan, data, annoy_index
    print(labels_part)
    # label,confi = confidence(result,labels_part)
    # log(servicebus_client ,f"best result for label: {label} with the confidence of {confi}", "ID REG - %s"%(CONTAINER_NAME))
    log(servicebus_client ,f"Time it took for variation {VARIATION} in total: {time.time()-total_time}", "ID REG - %s"%(CONTAINER_NAME))
    print("labels part: %s"%labels_part)
    print("labels part len: %s"%len(labels_part))
    part_number=part_name[4:-4]
    #send result
    queue_name_results="queueid"
    message_results="container-%s_variation-%s_part-%s_time-%s_"%(CONTAINER_NAME,VARIATION,part_number,str(time.time()-total_time))
    for label in result.keys():
        message_results+="%s-score-%s_"%(label, str(result[label]))
    send_message(servicebus_client,queue_name_results, message_results)
    del result
