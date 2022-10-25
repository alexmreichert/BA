
#############
#############
"""imports"""
#############
#############

#azure
from cmath import pi
import copy
import random as randomclass
from tkinter import N
from azure.storage.blob import ContainerClient
from azure.core import exceptions
from pandas import read_xml
import yaml
from azure.servicebus import ServiceBusClient, ServiceBusMessage, exceptions
from datetime import datetime

#open3d
import time
import open3d as o3d
import os
import numpy as np
from copy import deepcopy

#dataset creation
import h5py


###############
###############
"""functions"""
###############
###############
def rotation_matrix(axis):
    phi_x=-np.arctan([axis[1]/axis[2]])
    #print(f"phi_x= {phi_x*360/(np.pi*2)}")
    phi_y=-np.arctan([axis[0]/axis[2]])
    #print(f"phi_y= {phi_y*360/(np.pi*2)}")
    return rotation_matrix_3d(phi_x, phi_y)

def rotation_matrix_3d(phi_x, phi_y, phi_z=0):
    rotation_x=np.asarray([
        [1.0,0,0],
        [0,np.cos(phi_x),-np.sin(phi_x)],
        [0,np.sin(phi_x),np.cos(phi_x)]
    ],dtype='float64')
    rotation_y=np.asarray([
        [np.cos(phi_y),0,np.sin(phi_y)],
        [0,1.0,0],
        [-np.sin(phi_y),0,np.cos(phi_y)]],dtype='float64')
    rotation_z=np.asarray([
        [np.cos(phi_z),-np.sin(phi_z),0],
        [np.sin(phi_z),np.cos(phi_z),0],
        [0,0,1.0]],dtype='float64')
    return np.matmul(np.matmul(rotation_x,rotation_y),rotation_z).astype('float64')

def unit_vector(vector): return (vector/np.linalg.norm(vector)).astype('float64')

def add_sin_noise(pcd, variation):
    pcd_points=np.asarray(pcd.points)
    noise_x=0.05*np.random.rand(pcd_points.shape[0])*(np.sin(pcd_points[:,1]+np.sin(pcd_points[:,2])))
    noise_y=0.05*np.random.rand(pcd_points.shape[0])*(np.sin(pcd_points[:,0]+np.sin(pcd_points[:,2])))
    noise_z=0.05*np.random.rand(pcd_points.shape[0])*(np.sin(pcd_points[:,0]+np.sin(pcd_points[:,1])))
    sin_noise=np.c_[noise_x,noise_y,noise_z]
    pcd_points_with_noise=pcd_points-variation*sin_noise
    pcd_with_noise = o3d.geometry.PointCloud()
    pcd_with_noise.points=o3d.utility.Vector3dVector(pcd_points_with_noise)
    pcd_with_noise.normals = pcd.normals
    pcd_with_noise.colors = pcd.colors
    return pcd_with_noise

def add_noise(pcd, variation):
    pcd_points=np.asarray(pcd.points)
    pcd_points_with_noise=pcd_points-variation*0.005*np.random.rand(pcd_points.shape[0],pcd_points.shape[1])
    pcd_with_noise = o3d.geometry.PointCloud()
    pcd_with_noise.points=o3d.utility.Vector3dVector(pcd_points_with_noise)
    pcd_with_noise.normals = pcd.normals
    pcd_with_noise.colors = pcd.colors
    return pcd_with_noise

def RANSAC(pcd_ransac,d_threshold, paint_red=False):
    
    plane_model, inliers = pcd_ransac.segment_plane(distance_threshold=d_threshold*1.5, ransac_n=3, num_iterations=1000)
    pcd_object=pcd_ransac.select_by_index(inliers, invert=True)
    pcd_plane=pcd_ransac.select_by_index(inliers, invert=False)
    if paint_red: pcd_plane.paint_uniform_color([1,0,0])
    return pcd_object,pcd_plane,plane_model

def distance_point_plane(point, plane):     return abs(plane[0]*point[0]+plane[1]*point[1]+plane[2]*point[2]+plane[3])/np.linalg.norm(plane[:3])

def alpha_hull_pcd(pcd, number_of_points=10000):
    # tetra_time=time.time()
    pcd = pcd.uniform_down_sample(2)
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd) #bottleneck for larger points
    # mesh_time=time.time()
    # print(f"it took {mesh_time-tetra_time} seconds to creat tetra mesh with {len(np.asarray(pcd.points))}")
    mesh_alpha = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 10^1000000, tetra_mesh, pt_map)
    # sample_time=time.time()
    # print(f"it took {sample_time-mesh_time} seconds to creat alpha mesh")
    pcd_alpha = mesh_alpha.sample_points_poisson_disk(number_of_points=number_of_points, init_factor=2)
    # print(f"it took {time.time()-mesh_time} seconds to smaple alpha mesh")
    # o3d.visualization.draw_geometries([mesh_alpha,pcd_alpha])
    return pcd_alpha

def remove_plane(pcd,points_of_pcd,plane_model, center_point, slice_distance_factor):
    distance_to_center=distance_point_plane(center_point, plane_model)
    closest_point=center_point-distance_to_center*plane_model[:3]
    points_dot=np.dot(points_of_pcd-(np.ones((points_of_pcd.shape[0],1))*[center_point]),plane_model[:3])
    indices_slice=np.nonzero(points_dot>-slice_distance_factor*distance_to_center)
    pcd_view=pcd.select_by_index(indices_slice[0])

    return pcd_view,closest_point

def get_slices(pcd_alpha, pcd, slice_distance_factor=0, maxshapes=12, maxviews=4,vizualize=False):
    center_point=pcd.get_center()  
    if vizualize:
        sphere=o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0,1,0])
        sphere.translate(center_point)
        
        # o3d.visualization.draw_geometries([pcd,sphere])  
    points_of_pcd=np.asarray(pcd.points)
    pcd_views=[]
    plane_models=[]
    closest_points=[]
    pcd_object=pcd_alpha
    for _ in range(maxshapes):
        if len(np.asarray(pcd_object.points))<len(np.asarray(pcd_alpha.points))*0.2:break
        pcd_object,pcd_plane,plane_model = RANSAC(pcd_object, 0.0005)
        if np.dot(np.asarray(pcd_plane.points)[0]-center_point,plane_model[:3]) > 0: 
            plane_model=-plane_model
        plane_models.append(plane_model)
        del pcd_plane, plane_model
    del pcd_object

    print("got %s plane models"%(len(plane_models)))
    for count_plane_model, plane_model in enumerate(plane_models): #loop through planes find orthagonal
        pcd_view,closest_point=remove_plane(pcd,points_of_pcd,plane_model,center_point,slice_distance_factor)
        print("removed first slice for plane %s"%(count_plane_model))
        closest_points.append(closest_point)
        pcd_views.append(pcd_view)
        if vizualize:o3d.visualization.draw_geometries([pcd_view,sphere])

        #remove adjacent views 
        adjacent_planes_found=0
        pcd_view_prev=copy.deepcopy(pcd_view)
        del pcd_view
        for count_plane_compared, plane_compared in enumerate(plane_models):
            if adjacent_planes_found>=maxviews: break
            if abs(np.dot(plane_compared[:3],plane_model[:3]))<=0.01:
                pcd_view_prev_points=np.asarray(pcd_view_prev.points)
                pcd_view_prev,_=remove_plane(pcd_view_prev,pcd_view_prev_points,plane_compared,center_point,slice_distance_factor)
                pcd_views.append(pcd_view_prev)
                adjacent_planes_found+=1
                print("removed adjacent slice %s"%(adjacent_planes_found))
            print("compared plane %s with plane %s"%(count_plane_compared,count_plane_model))
    return pcd_views, plane_models, closest_points, center_point

def indexes_in_angle_range(phi,min_tmp,delta):
    min=min_tmp
    if min<0:min+=2*pi
    if min>2*np.pi:min-=2*pi*(np.floor(min/2*np.pi))
    max=min+delta
    indices_larger_than_min=np.nonzero(min<phi)[0]
    indices=np.zeros(len(phi))
    if max>2*np.pi:
        max-=2*pi
        indices_smaller_than_max=np.nonzero(max>phi)[0]
        indices=np.concatenate((indices_larger_than_min,indices_smaller_than_max))
    else: 
        indices_smaller_than_max=np.nonzero(max>phi)[0]
        indices=np.intersect1d(indices_smaller_than_max, indices_larger_than_min)
    return indices,max

def slice_pcd(pcds, plane_models, closest_points, center,N, phi_z_init=0,vizualize=False): 
    i=0
    pcds_slices=[]
    for i_pcd in range(len(pcds)):
        pcd=pcds[i_pcd]
        plane_model=plane_models[i_pcd]
        closest_point=closest_points[i_pcd]

        #vizualize the problem
        if vizualize:
            sphere=o3d.geometry.TriangleMesh.create_sphere(radius=1)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color([0,1,0])
            sphere.translate(center)
            
            o3d.visualization.draw_geometries([pcd,sphere])


        x_axis=unit_vector(np.dot([1,0,0],rotation_matrix(plane_model[:3])))
        y_axis=unit_vector(np.cross(x_axis,plane_model[:3]))
        points_raw=np.asarray(pcd.points)
        vector=points_raw-center
        dot_x    = np.dot(vector, x_axis)
        dot_y    = np.dot(vector, y_axis)
        r = np.sqrt(np.multiply(dot_x,dot_x)+np.multiply(dot_y,dot_y))
        phi_z_sin= np.arcsin(np.divide(dot_x,r))
        phi_z_cos= np.arccos(np.divide(dot_y,r))
        phi_z=np.zeros((len(points_raw),1))
        indices_cos_pos=np.nonzero(phi_z_cos<=np.pi/2)
        indices_cos_neg=np.nonzero(phi_z_cos>np.pi/2)
        indices_sin_pos=np.nonzero(phi_z_sin>=0)
        indices_sin_neg=np.nonzero(phi_z_sin<0)
        index_pos_pos=np.intersect1d(indices_cos_pos, indices_sin_pos)
        index_pos_neg=np.intersect1d(indices_cos_pos, indices_sin_neg)
        index_neg_pos=np.intersect1d(indices_cos_neg, indices_sin_pos)
        index_neg_neg=np.intersect1d(indices_cos_neg, indices_sin_neg)
        np.put(phi_z,index_pos_pos,phi_z_cos[index_pos_pos])
        np.put(phi_z,index_pos_neg,phi_z_sin[index_pos_neg])
        np.put(phi_z,index_neg_pos,phi_z_cos[index_neg_pos])
        np.put(phi_z,index_neg_neg,phi_z_cos[index_neg_neg]+np.pi/2)
        index_phi_z_neg=np.nonzero(phi_z<0)[0]
        np.put(phi_z,index_phi_z_neg,phi_z_sin[index_phi_z_neg]+2*np.pi)
        delta_phi_z=2*np.pi/N
        if phi_z_init<0: phi_z_init+2*np.pi
        phi_z_start=phi_z_init
        for n in range(N):
            indices,phi_z_start=indexes_in_angle_range(phi_z,phi_z_start,delta_phi_z)
            pcd_slice=pcd.select_by_index(indices)
            pcds_slices.append(pcd_slice)

            if vizualize:
                o3d.visualization.draw_geometries([pcd_slice,sphere])

    
    return pcds_slices

#    return sliced_pcds
def vizualize_in_matrix(pcds,width=1):
    box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcds[0].points)
    matrix_size=max(np.dot([1,0,0],box.get_extent()),np.dot([0,1,0],box.get_extent()))
    size_x= matrix_size*np.asarray([5,0,0])
    # print(f"x: {size_x}")
    size_y= matrix_size*np.asarray([0,5,0])
    # print(f"y: {size_y}")
    # print(f"parts: {len(pcds)}")
    pcds_temp=[]
    h=0
    w=0
    for i in range(len(pcds)):
        pcd=pcds[i]
        # print(f"x,y: {[w,h]}")
        pcd.translate(size_y*h)
        pcd.translate(size_x*w)
        w+=1
        if w>width-1:
            w=0
            h+=1
        pcds_temp.append(pcd)
    
    o3d.visualization.draw_geometries(pcds_temp)

def generate_pcd(service_bus_client, mesh, r, n, s, b,slice_variations, complete, file_name, print_times=False, sample_points=6000,alpha_number_of_sampled_points=4000, include_og=True, mode="reg",density=0.04187802201550127, radius_factor=0.01):
    log(service_bus_client, f"generating pcds of: {r, n, s, b,slice_variations} for {mode}")
    total_time=time.time()
    pcds_for_mesh=[]
    mesh.scale(0.001,[0,0,0])

    #poison disk sampling to create pcd
    final_alpha_size=sample_points
    if mode=="reg":
        final_alpha_size=int(np.ceil(density*(10e6)*mesh.get_surface_area()))

    t=time.time()
    pcd = mesh.sample_points_poisson_disk(number_of_points=alpha_number_of_sampled_points, init_factor=2)
    log(service_bus_client, f"File took {time.time()-t} seconds to create points cloud from {file_name}")
    #ball pivot to create mesh
    # if include_og: pcds_for_mesh.append(pcd)
    testing=True
    time_alpha=time.time()
    pcd_alpha = alpha_hull_pcd(pcd,alpha_number_of_sampled_points)
    log(service_bus_client, f"finished getting alpha hull pcd: {pcd_alpha} in {time.time()-time_alpha} seconds")

    for r_i in range(r):
        t_ball=time.time()
        r_factor=abs(1-r_i*np.random.rand(1)/r)
        # log(service_bus_client, "got radius factor %s"%(r_factor))
        radii = r_factor*np.asarray([radius_factor, radius_factor, radius_factor, radius_factor])
        alpha = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                                pcd, o3d.utility.DoubleVector(radii))
        # log(service_bus_client, f"File took {time.time()-t} seconds to create alpha shape from {file_name} and r value {r_i}")
        pcd_alpha_ball = alpha.sample_points_poisson_disk(number_of_points=final_alpha_size, init_factor=2)
        log(service_bus_client, f"File took {time.time()-t_ball} seconds to create {pcd_alpha_ball} from {file_name} and r value {r_factor*radius_factor}")
        pcds_for_mesh.append(pcd_alpha_ball)
    log(service_bus_client, f"PCD mesh size: {len(pcds_for_mesh)}")
    
    pcds_temp_views=deepcopy(pcds_for_mesh)
    for v_i in range(b):
        t=time.time()
        time_views=time.time()
        for pcd in pcds_temp_views:
            pcds_views, plane_models, closest_points, center = get_slices(pcd_alpha, pcd, slice_distance_factor=0.9-0.05*v_i*np.random.rand(1),maxshapes=s)
            pcds_for_mesh+=pcds_views
            
        log(service_bus_client, f"finished getting {len(pcds_views)} views in {time.time()-time_views} seconds")
        
        if not complete:
            for slice_variation in range(slice_variations):
                slice_var=3+slice_variation
                pcds_slices = slice_pcd(pcds_for_mesh,plane_models, closest_points,center,slice_var,2*np.pi*np.random.rand(1))
                testing=False
                pcds_for_mesh+=pcds_slices
                

        log(service_bus_client, f"File took {time.time()-t} seconds to create slices and alpha hull {file_name} and v_i value {v_i}")
    log(service_bus_client, "finished getting %s views"%(b))
    
    
    # pcds_alpha_temp=deepcopy(pcds_for_mesh)
    pcds_with_noise=[]
    pcds_temp=deepcopy(pcds_for_mesh)
    # print(pcds_temp)
    t_n=time.time()
    new_n=n
    add_sinus=False
    if n==64: 
        new_n=32
        add_sinus=True

    for n_i in range(new_n):
        iteration=0
        for pcd in pcds_temp:
            phi_x, phi_y, phi_z = np.pi*2*np.random.rand(3)
            random_rotation_matrix = rotation_matrix_3d(phi_x, phi_y, phi_z)
            pcd.rotate(random_rotation_matrix)
            pcd.translate(-pcd.get_center())
            iteration+=1
            
            pcd_with_noise=add_noise(pcd,1-n_i/(new_n-1))
            # o3d.visualization.draw_geometries([pcd_with_noise])
            pcds_for_mesh.append(pcd_with_noise)
            if add_sinus:
                pcd_with_sin_noise=add_sin_noise(pcd,1-n_i/(new_n-1))
                # o3d.visualization.draw_geometries([pcd_with_sin_noise])
                pcds_for_mesh.append(pcd_with_sin_noise)
    log(service_bus_client, f"File took {time.time()-t_n} seconds to add noise {file_name} and times{n}")
    # pcds_temp1=deepcopy(pcds_for_mesh)
    # print(pcds_temp1)


    
    log(service_bus_client, f"File took {time.time()-total_time} seconds in total to create {len(pcds_for_mesh)} variations")
    return pcds_for_mesh

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

def save_to (pcds, dir, file_name, print_stuff):
    i=0
    for pcd in pcds:
        write_name=os.path.join(dir,f"{file_name[:-4]}_pcd_{i}.pcd")
        if print_stuff: print(f"saving {write_name}")
        o3d.io.write_point_cloud(write_name, pcd)
        i+=1

#azure
def download(connection_string, container_name, save_path, servicebus_client):
    labels=[]
    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    if not container_client.exists(): 
        log(servicebus_client, "wrong connection_string")
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
            # container_client.delete_blob(blob, delete_snapshots="include")
            file.write(data)
            labels.append(blob.name)
            file.close()
    log(servicebus_client, f"saving locally: {labels}")
    return labels

def get_stl(config, servicebus_client, pipeline_name, download_pipename, MAX_WAIT_TIME):
    changestr=[]
    labels=[]
    mode="reg"
    received=False
    #wait for message and download files
    with servicebus_client:
        # get the Subscription Receiver object for the subscription 
        queue_name=  config['queue_name']+pipeline_name
        receiver = servicebus_client.get_queue_receiver(queue_name=queue_name, max_wait_time=30)
        with receiver:
            messages = receiver.receive_messages(max_wait_time=1)
            i=1
            while ((not messages)) and i<MAX_WAIT_TIME: 
                log(servicebus_client, f"waiting for upload completion for {i} of {MAX_WAIT_TIME} sec.")
                time.sleep(1)
                messages = receiver.receive_messages(max_wait_time=1)
                i+=1
            if messages:
                for message in messages:
                    msg_split=str(message).split("_")
                    mode=msg_split[0]              
                    if not ((mode=="ml") or (mode=="reg")):
                        log(servicebus_client, f"message kinda wierd ngl -_-: {message}")
                        receiver.complete_message(message)
                        continue
                    else: 
                        received=True
                        changestr = msg_split 
                        log(servicebus_client, "changestr: %s"%(changestr))
                        receiver.complete_message(message)
            else: 
                log(servicebus_client, "no message next run")
        if received:
            container_name = config['container_name']+download_pipename
            labels = download(config['azure_storage_connectionstring'],container_name, os.path.join(os.path.dirname(os.path.abspath(__file__)),config['save_folder']),servicebus_client)
            log(servicebus_client, f"finished download from container: {container_name}")          
    return labels, received, mode, changestr

def load_config(dir_config):
    with open(os.path.join(dir_config, "config.yaml"), "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)

def upload_pcds(pcds,label, changestr, connection_string, container_name,queue_name, servicebus_client, tmp_path="save", print_stuff=False, upload_files=True):
    file_names=[]
    container_client = ContainerClient.from_connection_string(connection_string, container_name)   
    if not container_client.exists():
        log(servicebus_client, f"creating new container with {container_name}")
        try:
            container_client.create_container()
        except exceptions.ResourceExistsError:
            if print_stuff: print("waiting for deletion")
            time.sleep(20)
            container_client.create_container()
    i=0      
    names=[]
    for pcd in pcds:
        name_str='_'.join(changestr)+label
        write_name=f"{name_str}_var-{i}_.pcd"
        file_names.append(write_name)
        write_path=os.path.join(tmp_path,write_name)
        try:
            o3d.io.write_point_cloud(write_path, pcd)
            if upload_files:
                blob_client = container_client.get_blob_client(write_name)
                with open(write_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
            names.append(write_name)
            # send_completion_message(servicebus_client,queue_name,container_name,write_name)
        except FileNotFoundError: 
            log(servicebus_client, f"error not found file: {write_path}")
        i+=1
    log(servicebus_client, f"{len(names)} point clouds where uploaded for {label}")
    return file_names

def send_completion_message(servicebus_client,queue_name, container_name,label):
    sender = servicebus_client.get_queue_sender(queue_name=queue_name)
    with sender:  
          #'done--{container_name}--{Bauteil-19.stl}'
        message=f"done--{container_name}--{label}"   
        sender.send_messages(ServiceBusMessage(message))

def get_files(dir):
    #print(os.scandir(dir))

    with os.scandir(dir) as entries:
        for entry in entries:
            if (entry.is_file()) and ((entry.name.endswith(".stl") or entry.name.endswith(".ply"))):
                #print(entry.name)
                yield entry # turns into a generator function

def edit_config(changestr, variation ,path):
    with open(os.path.join(path,"config.yaml")) as f:
        list_doc = yaml.safe_load(f)
    
    #ml_radius-1_noise-1_view-1_batch-1_size-1_
    #reg_points-0_radius-0_feature-0_noise-0_
    mode=changestr[0]
    list_doc["mode"] = mode

    if mode=="reg":
        for str in changestr[1:]:
            if str=="": continue
            key=str.split("-")[0]
            var=str.split("-")[1]
            if key=="radius": list_doc["radius_factor"]=variation[key][int(var)]
            if key=="points": list_doc["points"]=variation[key][int(var)]
        list_doc["r"]=1
        list_doc["n"]=0
        list_doc["b"]=0
        list_doc["s"]=0
        list_doc["slice_variations"] = 0

    if mode=="ml":
        for str in changestr[1:]:
            if str=="": continue
            key=str.split("-")[0]
            var=str.split("-")[1]
            if key=="radius": list_doc["r"]=variation[key][int(var)]
            if key=="noise": list_doc["n"]=variation[key][int(var)]
            if key=="view": list_doc["b"]=variation[key][int(var)] #check
        list_doc["radius_factor"]=0.01
        list_doc["points"]=8000
        list_doc["s"]=6
        list_doc["slice_variations"] = 0
        list_doc["complete"] = "True"
    
    

    print(f"editing config file to: {list_doc} ")
    with open(os.path.join(path,"config.yaml"), "w") as f:
        yaml.dump(list_doc, f)

def log(service_bus_client,log,sender_info="IMG_GEN-"):
    try:
        sender = service_bus_client.get_queue_sender(queue_name="log")
        message=sender_info+str(datetime.now())+": "+log
        with sender:  
            sender.send_messages(ServiceBusMessage(message))
            print("log: %s"%(message))
    except exceptions.MessageSizeExceededError:
        print("queue full: %s"%(log))

#dataset
def create_dataset(file_names, variation, save_path, changestr, service_bus_client, label_names):
    #load setting
    point_length=2048
    file_names_shuffled=copy.deepcopy(file_names)
    randomclass.shuffle(file_names_shuffled)
    for str in changestr:
        #ml_radius-1_noise-1_view-1_batch-1_size-1_
        str_split=str.split("-")
        if str_split[0]=="size":
            point_length=variation[str_split[0]][int(str_split[1])]
    test_portion=0.2
    chunks=[]
    chunk_size=5000
    dataset_names=[]
    for i_file in range(0,len(file_names_shuffled),chunk_size):
        chunks.append(file_names_shuffled[i_file:i_file+chunk_size])

    for idx_chunk,chunk in enumerate(chunks):     
        #dataset size
        size_dataset=len(chunk)
        labels_to_prune=np.zeros((size_dataset))

        #initiate np array for points and labels
        points=np.zeros((size_dataset,point_length,3)) #shape: size_dataset x point_length x 3
        labels=np.zeros((size_dataset)) #shape: size_dataset
        
        #load data
        counter=0
        for file in chunk:
            #load points
            file_path=os.path.join(save_path,file)
            pcd = o3d.io.read_point_cloud(file_path)
            pcd_points=np.asarray(pcd.points)

            #randomsampling
            if len(pcd_points)<point_length:
                size_dataset-=1
                continue
            idx=np.arange(len(pcd_points))
            np.random.shuffle(idx)
            idx = idx[:point_length]
            points_sampled=pcd_points[idx,:]
            points[counter,:,:]=points_sampled
            file_label=file.split("_")[-3]
            labels[counter]=label_names.index(file_label)
            counter+=1
        points=points[:size_dataset,:,:] #shape: size_dataset x point_length x 3
        labels=labels[:size_dataset]
        #split dataset in to test and train
        idx_dataset=np.arange(size_dataset)#idx 
        np.random.shuffle(idx_dataset)#shuffle
        idx_cutoff=int(np.ceil((size_dataset*(1-test_portion))))#idx_of_idx  cut off
        idx_train=idx_dataset[:idx_cutoff]#idx for testing
        idx_test=idx_dataset[idx_cutoff:]#idx for training
        train_points=points[idx_train,:,:]#train points
        test_points=points[idx_test,:,:]#test points#
        train_labels=labels[idx_train]#train labels
        test_labels=labels[idx_test]#test labels

        log(servicebus_client, f"chunk {idx_chunk} - points train: {train_points.shape}")
        log(servicebus_client, f"chunk {idx_chunk} -points test: {test_points.shape}")
        log(servicebus_client, f"chunk {idx_chunk} -labels train: {train_labels.shape}")
        log(servicebus_client, f"chunk {idx_chunk} -labels test: {test_labels.shape}")

        ##remove BIAS
        try:
            sets=[[train_labels,train_points],[test_labels,test_points]]#loop through train and test set
            new_sets=[]
            for iter_labels, iter_points in sets:
                unique, count = np.unique(iter_labels,return_counts=True) #find amount count of labels 
                prune_len=np.amin(count)#prune length = smallest label
                log(servicebus_client, f"chunk {idx_chunk} -prune_len: {prune_len}")
                new_size=len(unique)*prune_len#new datasetsize
                new_labels=np.zeros((new_size))
                new_points=np.zeros((new_size,point_length,3))
                log(servicebus_client, f"chunk {idx_chunk} -new_points: {new_points.shape}")
                i=0
                for label in unique:#for each unique label 
                    index_label = np.where(iter_labels == label)[0]#get all index for each unqiue label
                    new_labels[i*prune_len:(i+1)*prune_len]=iter_labels[index_label[:prune_len]]#prune length and save to new length
                    new_points[i*prune_len:(i+1)*prune_len,:,:]=iter_points[index_label[:prune_len],:,:]
                    i+=1
                index_shuffle=np.arange(new_size)
                log(servicebus_client, f"chunk {idx_chunk} -new_points: {new_points.shape}")
                np.random.shuffle(index_shuffle)
                new_labels=new_labels[index_shuffle]
                new_points=new_points[index_shuffle,:,:]
                new_sets.append([new_labels,new_points])
            [[train_labels,train_points],[test_labels,test_points]] =  new_sets
        except Exception as e:
            log(servicebus_client, f"Error 561: {e}")
        log(servicebus_client, f"chunk {idx_chunk} -points train: {train_points.shape}")
        log(servicebus_client, f"chunk {idx_chunk} -points test: {test_points.shape}")
        log(servicebus_client, f"chunk {idx_chunk} -labels train: {train_labels.shape}")
        log(servicebus_client, f"chunk {idx_chunk} -labels test: {test_labels.shape}")

        #create h5 file
        file_name='_'.join(changestr)
        train_name='%strain_%s.h5'%(file_name,idx_chunk)
        dataset_names.append(train_name)
        train_path=os.path.join(save_path,train_name)
        f = h5py.File(train_path, 'w')
        dataset_points=f.create_dataset(name="data",shape=train_points.shape,dtype=np.dtype("float32"),data=train_points)
        dataset_points=f.create_dataset(name="label",shape=train_labels.shape,dtype=np.dtype(int),data=train_labels)
        f.close()
        test_name='%stest_%s.h5'%(file_name,idx_chunk)
        dataset_names.append(test_name)
        test_path=os.path.join(save_path,test_name)
        f = h5py.File(test_path, 'w')
        dataset_points=f.create_dataset(name="data",shape=test_points.shape,dtype=np.dtype("float32"),data=test_points)
        dataset_points=f.create_dataset(name="label",shape=test_labels.shape,dtype=np.dtype(int),data=test_labels)
        f.close()
    return dataset_names

def upload_dataset(dataset_names, path, connection_string, container_name, queue_name, servicebus_client):
    container_client = ContainerClient.from_connection_string(connection_string, container_name)   
    if not container_client.exists():
        log(servicebus_client, f"creating new container with {container_name}")
        try:
            container_client.create_container()
        except exceptions.ResourceExistsError:
            log(servicebus_client, "waiting for deletion")
            time.sleep(20)
            container_client.create_container()

    for dataset_name in dataset_names:
        read_path=os.path.join(path, dataset_name)
        try:
            blob_client = container_client.get_blob_client(dataset_name)
            with open(read_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
                log(servicebus_client, f"{dataset_name} uploaded")
            send_completion_message(servicebus_client,queue_name,container_name,dataset_name)
            os.remove(read_path)
        except FileNotFoundError: 
            log(servicebus_client, f"error not found file: {dataset_name}")

#############
#############
"""execute"""
#############
#############

#experiment values
variation_NN={
    "radius":[1,2],
    "noise": [16,64],
    "view": [1,16],
    "batch": [16,64],
    "size": [1024,2048],
}

variation_reg={
    "points":[2048,4096],
    "radius": [0.01,0.02],
    "feature": [0.001,0.003],
    "noise": [0.0001,0.0004],
}

pipename_local_to_data_prep = "lmdp"
pipename_data_prep_to_local= "dplm"
pipename_data_prep_to_ID_NN= "dpin"
pipename_data_prep_to_ID_reg= "dpir"
pipename_img_gen_to_train_reg= "imtr"
pipename_img_gen_to_train_ML= "imtm"
pipename_local_to_img_gen= "lmim"


config=load_config(os.path.dirname(os.path.abspath(__file__)))
print_stuff=True
pipename_img_gen_to_train=pipename_img_gen_to_train_reg

servicebus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connectionstring'])

max_runs=config['max_runs']
run=0
total_time=time.time()
while run<max_runs:
    log(servicebus_client, f"-----------------------Running {max_runs-run} runs more---------------------------------")
    t_run=time.time()
    run+=1
    labels, recieved_mes, mode, changestr=get_stl(config, servicebus_client, pipename_local_to_img_gen, pipename_local_to_img_gen, config['wait_time'])
    variation=variation_reg
    if mode=="reg":
        pipename_img_gen_to_train=pipename_img_gen_to_train_reg
        variation=variation_reg
    else:
        pipename_img_gen_to_train=pipename_img_gen_to_train_ML
        variation=variation_NN
    
    edit_config(changestr,variation,os.path.dirname(os.path.abspath(__file__)))
    config=load_config(os.path.dirname(os.path.abspath(__file__)))


    log(servicebus_client, f"got following files: {labels}")
    #meshes
    meshes=[]
    file_names=[]
    load_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), config['save_folder'])
    save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"save")

    found_stl= False
    dir_list=os.listdir(load_path)
    for file in dir_list:
        if file.endswith(".stl"):
            file_names.append(file[:-4])
            file_path=os.path.join(load_path,file)
            mesh_from_file = o3d.io.read_triangle_mesh(file_path)
            os.remove(file_path)
            mesh_from_file.compute_vertex_normals()
            mesh_from_file.paint_uniform_color([0.6, 0, 0])
            meshes.append(mesh_from_file)
            found_stl = True
    log(servicebus_client, f"loaded and deleted files: {file_names}")
    log(servicebus_client, f"message found: {recieved_mes}, files found: {found_stl}")
    if not recieved_mes and not found_stl: 
        continue
    else: run-=1
    if len(labels)<=len(file_names): labels=file_names
    time_generation=time.time()
    i=0
    all_files=[]
    for mesh in meshes:
        time_mesh=time.time()
        complete=True
        complete=   config['complete']=="True"
        include_og= config['mode']=="ml"
        pcds=[]
        pcds=generate_pcd(servicebus_client, mesh, config['r'], config['n'], config['s'], config['b'], config['slice_variations'], complete, file_names[i], print_times=print_stuff, sample_points=config['points'],alpha_number_of_sampled_points=config["alpha_number_of_sampled_points"], include_og=include_og, mode=config["mode"],density=config["density"], radius_factor=config["radius_factor"])
        #upload pcds
        container_name = config['container_name']+pipename_img_gen_to_train
        queue_name= config['queue_name']+pipename_img_gen_to_train
        #vizualize_in_matrix(pcds,width=4)
        # o3d.io.write_point_cloud(os.path.join(save_path,"generatedpart.pcd"), pcds[0])
        file_names_pcd=upload_pcds(pcds,labels[i], changestr, config['azure_storage_connectionstring'], container_name,queue_name,servicebus_client,tmp_path=save_path, upload_files=False)
        all_files+=file_names_pcd
        
        average_time_took=(time.time()-time_generation)/(i+1)
        total_time_pred=average_time_took*len(meshes)
        log(servicebus_client, "finished mesh %s in %s seconds - %s of %s or %s s of %s s"%(labels[i],time.time()-time_mesh,i+1,len(meshes),average_time_took*(i+1),total_time_pred))
        i+=1
    log(servicebus_client, f"------------------Run took: {time.time()-t_run} seconds---------------------------------")
    log(servicebus_client, f"------------------Generated: {len(all_files)}")
    if mode=="ml": #create h5 database and upload 
        log(servicebus_client, "labels: %s"%(labels))
        dataset_names =create_dataset(all_files, variation,save_path, changestr,servicebus_client,labels)
        container_name = config['container_name']+pipename_img_gen_to_train
        queue_name= config['queue_name']+pipename_img_gen_to_train
        upload_dataset(dataset_names, save_path,config["azure_storage_connectionstring"],container_name, queue_name, servicebus_client)
    #upload h5
        
    #delete all files
    for file in all_files:
        file_path=os.path.join(save_path,file)
        os.remove(file_path)
    log(servicebus_client, "removed %s"%(all_files))
log(servicebus_client, f"Time it took in total: {time.time()-total_time}")