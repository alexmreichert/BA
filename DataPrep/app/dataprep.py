###############
"""IMPORTS"""
###############
#azure storage and messaging
from turtle import color, width
from azure.servicebus import ServiceBusClient, ServiceBusMessage, exceptions
from azure.storage.blob import ContainerClient
from azure.core import exceptions

#utils
import copy
import os
import glob
import time
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

#Point Cloud calculations
import numpy as np
import open3d as o3d
import pickle
from scipy.optimize import minimize


###############
"""FUNCTIONS"""
###############
#config
def load_config(dir_config):
    with open(os.path.join(dir_config, "config.yaml"), "r") as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)

#util functions
def norm(vector):
    return (vector/np.linalg.norm(vector)).astype("float64")

def coordinate_system(normal,center):
    N_z=normal
    N_x=norm(np.cross(N_z,[1,0,0]))
    N_y=norm(np.cross(N_z,N_x))
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere.translate(center)
    sphere.compute_triangle_normals()

    #axis
    points=[
        center,
        center+N_z,
        center+N_y,
        center+N_x,
    ]

    lines=[
        [0,1],
        [0,2],
        [0,3],
    ]
    line_set=o3d.geometry.LineSet()
    line_set.points= o3d.utility.Vector3dVector(points)
    line_set.lines= o3d.utility.Vector2iVector(lines)

    return sphere, line_set

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

def create_spheres_and_lines(origin, line_models,color=[0,0,0],color_origin=[1,0,0]):
    points=[origin]
    lines=[]
    spheres=[]
    sphere_origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere_origin.compute_vertex_normals()
    sphere_origin.paint_uniform_color(color_origin)
    sphere_origin.translate(origin)
    spheres.append(sphere_origin)

    for line_model in line_models:
        point, distance=distance_point_to_line(origin, line_model)
        #print(f"distance: {distance}")
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        sphere.translate(point)
        spheres.append(sphere)

        points.append(point)
        lines.append([0,len(points)-1])
    
    

    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return spheres, line_set, points[1:]

def create_spheres(points,color=[0,0,1]):
    spheres=[]
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        sphere.translate(point)
        spheres.append(sphere)
    return spheres

#segmentation
def DBSCAN(pcd, d_threshold):
    labels = np.array(pcd.cluster_dbscan(eps=d_threshold*3, min_points=10))
    candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)] # find all amount of points for all categories
    best_candidate=int(np.unique(labels)[np.where(candidates== np.max(candidates))[0]])
    seg=pcd.select_by_index(list(np.where(labels!=best_candidate)[0]))
    pcd_clean=pcd.select_by_index(list(np.where(labels== best_candidate)[0]))
    seg.paint_uniform_color([1,0,0])
    return seg, pcd_clean

def RANSAC(pcd,d_threshold, paint_red=False):
    plane_model, inliers = pcd.segment_plane(distance_threshold=d_threshold*1.5, ransac_n=3, num_iterations=1000)
    pcd_object=pcd.select_by_index(inliers, invert=True)
    pcd_plane=pcd.select_by_index(inliers, invert=False)
    if paint_red: pcd_plane.paint_uniform_color([1,0,0])
    return pcd_object,pcd_plane,plane_model

def cylinder_removal(pcd, center, N_z, radius=0.4, min_z=-0.005, max_z=0.2):
    rotation=rotation_matrix(N_z)
    N_x=norm(np.dot(rotation, [1,0,0]))
    N_y=norm(np.cross(N_x,N_z))
    N_z=norm(N_z) 
    
    center=np.reshape(center,(1,3))
    points=np.asarray(pcd.points)
    center_points=np.ones((len(points),1))*center
    
    indexes=np.arange(len(points))

    # print(center_points)
    diff=points-center_points
    diff_x = np.dot(diff,N_x)
    diff_y = np.dot(diff,N_y)
    diff_z = np.dot(diff,N_z)
    point_rad=np.sqrt(np.square(diff_x)+np.square(diff_y))

    inliers=indexes[(diff_z>=min_z)& (point_rad<radius) & (diff_z<=max_z)]
    # print(point_rad[inliers])
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([1,0,0])
    # o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud])
    return inliers, inlier_cloud, outlier_cloud

def list_of_points_on_plane(plane_model, N):
    #print(f"plane model {plane_model}")
    seed=np.random.rand(N,2)
    #print(f" random seed: {seed}")
    z =np.reshape(np.asarray(-((plane_model[0]*seed[:,0])+(plane_model[1]*seed[:,1])+plane_model[3])/plane_model[2]), (N,1))
    #print(f"z axis: {z.shape}")
    plane_points=np.c_[seed, z]
    #print(f"plane points: {plane_points}")
    return plane_points

#abstraction to 2d
def distance_point_plane(point, plane):     return abs(plane[0]*point[0]+plane[1]*point[1]+plane[2]*point[2]+plane[3])/np.linalg.norm(plane[:3])

def list_of_points_on_plane(plane_model, N):
    #print(f"plane model {plane_model}")
    seed=np.random.rand(N,2)
    #print(f" random seed: {seed}")
    z =np.reshape(np.asarray(-((plane_model[0]*seed[:,0])+(plane_model[1]*seed[:,1])+plane_model[3])/plane_model[2]), (N,1))
    #print(f"z axis: {z.shape}")
    plane_points=np.c_[seed, z]
    #print(f"plane points: {plane_points}")
    return plane_points

def line_equation(plane_model_1, plane_model_2):
    A_1,B_1,C_1,D_1=plane_model_1
    A_2,B_2,C_2,D_2=plane_model_2
    N=np.asarray([
            -((B_1/A_1)*((A_2*D_1-D_2*A_1)/(B_2*A_1-A_2*B_1))+(D_1/A_1)),# -((D_1/A_1)+((B_1/A_1)*(D_2-(A_2*D_1/A_1))/(B_2-(C_2*B_1/A_1)))),
            (A_2*D_1-D_2*A_1)/(B_2*A_1-A_2*B_1),# (D_2-(A_2*D_1/A_1))/(B_2-(C_2*B_1/A_1)),
            0# 0
        ])
    v_abs=np.asarray([
            -((B_1/A_1)*((A_2*C_1-C_2*A_1)/(B_2*A_1-A_2*B_1))+(C_1/A_1)),# -((C_1/A_1)+(B_1/A_1)*(C_2-(C_1*A_2/A_1))/(B_2-(C_2*B_1/A_1))),
            (A_2*C_1-C_2*A_1)/(B_2*A_1-A_2*B_1),# (C_2-(C_1*A_2/A_1))/(B_2-(C_2*B_1/A_1)),
            1,# 1
        ])
    # print(f"v in function: {v_abs/np.linalg.norm(v_abs)}")
    return np.asarray(
        [N,
        v_abs]#/np.linalg.norm(v_abs)]
    )

def list_of_points_on_line(line_model,N):
    seed=-np.random.rand(N,1)
    # print(np.multiply(np.reshape(line_model[0],(1,3))[0],np.ones((N,1))))
    # print(np.multiply(np.reshape(line_model[1],(1,3))[0],seed))
    return np.multiply(np.reshape(line_model[0],(1,3))[0],np.ones((N,1)))+np.multiply(np.reshape(line_model[1],(1,3))[0],seed)

def distance_point_to_line(point, line_model):
    N,v=line_model
    # print(f"N: {N}")
    # print(f"v: {v}")
    # print(f"point: {point}")
    t=(np.dot(v,point)-np.dot(N,v))/np.dot(v,v)
    return N+np.multiply(t,v), np.linalg.norm(N+np.multiply(t,v)-point)

def difference_between_distances(point_init, args):
    distances=[]
    difference=0
    (line_models,(floor_plane_model_avrg,origin))=args
    N_1=np.asarray(floor_plane_model_avrg[:3]/np.linalg.norm(floor_plane_model_avrg[:3]))
    point = point_init-N_1*distance_point_plane(point_init,floor_plane_model_avrg)
    if distance_point_plane(point,floor_plane_model_avrg)!=0:point = point_init+N_1*distance_point_plane(point_init,floor_plane_model_avrg)
    for line_model in line_models:
        # print(f"point: {point}")
        # N,v=line_model
        # print(f"N,v: {N,v}")
        point_on_line, distance=distance_point_to_line(point, line_model)
        #print(f"distance: {distance}")
        distances.append(distance)
        if np.dot(point_on_line-point,point_on_line-origin)<=0: 
            difference+=1000
            #print(f"difference before loop: {difference}")
    for distance1 in distances:
        for distance2 in distances:
            difference+=abs(distance2-distance1)
            #print(f"difference in loop: {difference}")
    return difference

def find_normal_line(origin, line_models,rotation):
    angle_differences=[]
    for line_model in line_models:
        N,v =line_model
        y_axis=norm(np.dot([0,1,0],rotation))
        line=norm(v)
        #print(np.dot(y_axis,point_to_orgin))
        angle_differences.append(np.cos(np.dot(y_axis,v)))
    # print(angle_differences)
    # print(min(angle_differences))
    return angle_differences.index(min(angle_differences))

def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = (np.vstack([x**2, x*y, y**2]).T).astype('float64')
    D2 = (np.vstack([x, y, np.ones(len(x))]).T).astype('float64')
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    # print(f"S1: {S1}")
    # print(f"S2: {S2}")
    # print(f"S3: {S3}")
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi

def points_on_plane_to_2D(points_3D,N_1):
    rotation=rotation_matrix(N_1)#np.linalg.inv(rotation_matrix(N_1))
    x_axis=norm(np.dot(rotation, [1,0,0]))
    y_axis=norm(np.cross(x_axis,N_1))#####
    z_axis=N_1
    x=np.dot(points_3D, x_axis) # Nx1 of 3D points Nx3
    y=np.dot(points_3D, y_axis)
    z=np.dot(points_3D, z_axis)
    return np.c_[x,y,z]

def get_ellipse_pts(params, npts=200, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

def get_circle_pts(params, npts=200, tmin=0, tmax=2*np.pi):
    x0, y0, radius = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + radius * (np.cos(t) )
    y = y0 + radius * (np.sin(t) )
    return x, y

def points_on_xy_to_3d(points_2d, N_1, points_3d_sample=None):
    rotation=rotation_matrix(N_1)
    x=np.dot(points_2d, [1,0,0])
    y=np.dot(points_2d, [0,1,0])
    z=np.dot(points_2d, [0,0,1])
    x_axis=norm(np.dot(rotation, [1,0,0]))
    y_axis=norm(np.cross(x_axis,N_1))
    z_axis=N_1
    point_3d_calc= np.multiply(np.reshape(x,(len(x),1)),np.reshape(x_axis,(1,len(x_axis))))+np.multiply(np.reshape(y,(len(y),1)),np.reshape(y_axis,(1,len(y_axis))))+np.multiply(np.reshape(z,(len(z),1)),np.reshape(z_axis,(1,len(z_axis))))
    if points_3d_sample: print(f"Difference: {np.linalg.norm(point_3d_calc-points_3d_sample)}")
    return point_3d_calc


    points_raw=np.asarray(pcd.points)
    x_axis = norm(np.dot([1,0,0],rotation_matrix(N_1)))
    y_axis = norm(np.cross(x_axis,N_1))
    z_axis = N_1
    vector=points_raw-origin
    vector_x    = np.dot(vector, x_axis)
    vector_y    = np.dot(vector, y_axis)
    vector_z    = np.dot(vector, z_axis)
    index_xy=np.nonzero((np.multiply(vector_x,vector_x)+np.multiply(vector_y,vector_y))<radius**2)[0]
    index_z=np.nonzero(vector_z>min_z)[0]
    inliers=np.intersect1d(index_xy, index_z)
    inlier_cloud = pcd.select_by_index(inliers)
    return inliers, inlier_cloud

def transform_around_axis(phi_z, N_1,center, rotation):
    x_axis=norm(np.dot(rotation, [1,0,0]))
    y_axis=norm(np.cross(x_axis,N_1))
    z_axis=N_1
    P=np.c_[x_axis,y_axis,z_axis]
    P_inv=np.linalg.inv(P)
    rotation_matrix_around_z=rotation_matrix_3d(0,0,phi_z)
    print(f"rotated {phi_z*360/(2*np.pi)}°")
    rotation_matrix_tmp = np.matmul(np.matmul(P,rotation_matrix_around_z),P_inv)
    translation_transf=np.vstack([np.c_[np.eye(3),center],[0,0,0,1]])
    translation_neg_transf=np.vstack([np.c_[np.eye(3),-center],[0,0,0,1]])
    rotation_transf= np.vstack([np.c_[rotation_matrix_tmp,np.zeros(3)],[0,0,0,1]])
    return np.matmul(np.matmul(translation_transf, rotation_transf),translation_neg_transf)

#distortion removal
def correction_ellipse(points, N_1, center, circle_param, eclipse_param):
    _, _, ap, bp, e, phi = eclipse_param    #unload parameters eclipse
    _, _, ac=circle_param                   #unload parameters circle
    x0,y0,z0=center                         #get center points
    correction_factor=min(ac/ap,ac/bp)      #calculate of compression factor need
    phi=np.pi-phi
    rotation=rotation_matrix(N_1)   #calc new coordinate system
    x_axis_temp=norm(np.dot(rotation, [1,0,0]))
    y_axis_temp=norm(np.cross(x_axis_temp,N_1))
    z_axis=norm(N_1) 
    x_axis = x_axis_temp*np.cos(phi)-y_axis_temp*np.sin(phi)#factor in phi
    y_axis = x_axis_temp*np.sin(phi)+y_axis_temp*np.cos(phi)
    rot=np.c_[x_axis, y_axis, z_axis]#create transformation matrix for coordinate system
    rot_inv=np.linalg.inv(rot)#invert 

    vector=points-center #transform center to base of coordinate
    dot_x    = np.dot(vector, x_axis)
    dot_y    = np.dot(vector, y_axis)
    dot_z    = np.dot(vector, z_axis)

    dot_x_new=dot_x*correction_factor#apply correction factor to x
    new_points=np.c_[dot_x_new,dot_y,dot_z]#combine to new points in 
    new_points.flatten()
    new_points=np.dot(new_points, rot_inv)#convert to old coordinate system 
    new_points+=center#translate back to old base
    return new_points

def correction_y(points,turntable_normal,plane_model,center):
    plane_normal=plane_model[:3]

    rotation=rotation_matrix(turntable_normal)   #calc new coordinate system
    x_axis_temp=norm(np.dot(rotation, [1,0,0]))
    y_axis_temp=norm(np.cross(x_axis_temp,turntable_normal))
    z_axis=norm(turntable_normal) 
    plane_norm_x=np.dot(x_axis_temp,plane_normal)#x components of plane normal
    plane_norm_y=np.dot(y_axis_temp,plane_normal)#y components of plane normal
    phi=np.arctan(plane_norm_x/plane_norm_y)# rad to turn to fit y with normal
    x_axis = norm(x_axis_temp*np.cos(phi)-y_axis_temp*np.sin(phi))#rotate that y is aligned with normal
    y_axis = norm(x_axis_temp*np.sin(phi)+y_axis_temp*np.cos(phi))
    rot=np.c_[x_axis, y_axis, z_axis]#create transformation matrix for coordinate system
    rot_inv=np.linalg.inv(rot)#invert 

    new_points=points-center #transform center to base of coordinate
    new_points=np.dot(new_points,rot)#transform points to new coordinates   
    plane_normal=np.dot(plane_normal,rot)#transform normal of surface to new coordinate system
    turntable_normal=np.dot(turntable_normal,rot)#transform normal of turntable to new coordinate system
    
    alpha=np.arccos(np.dot(plane_normal,turntable_normal)/(np.linalg.norm(plane_normal)*np.linalg.norm(turntable_normal)))#calculate angle between both normals
    parameter=np.tan((np.pi/2)-alpha)#calculate parameter scalar
    z=new_points[:,2] # z-axis 
    delta_points=parameter*z#delta points = parameter * z 
    new_y=new_points[:,1]+delta_points#new points
    new_points=np.c_[new_points[:,0],new_y,new_points[:,2]]
    new_points=np.dot(new_points,rot_inv)#transform back to old coordinate system
    new_points=new_points+center
    return new_points

#azure
def download_files(part_number, container_client, prefix="part"):
    part_name="%s%s_scan"%(prefix,part_number)
    print(part_name)
    if len(glob.glob("%s/%s*.pcd"%(temp_dir,part_name)))==0:

        if not container_client.exists(): 
                print("wrong connection_string")
        else:
            blobs=container_client.list_blobs()
            for blob in blobs:
                blob_name=blob.name
                if blob_name.startswith(part_name):
                    StorageStreamDownloader = container_client.download_blob(blob)
                    filename=os.path.join(temp_dir,blob_name)
                    try:
                        file = open(filename, 'wb')
                    except FileNotFoundError:
                        os.mkdir(temp_dir)
                        file = open(filename, 'wb')
                    data=StorageStreamDownloader.readall()
                    print(f"saving locally blob: {blob_name}")
                    file.write(data)
                    file.close()

def receive_upload_message(service_bus_client,queue_name,config):
    i=0
    MAX_WAIT_TIME=config["wait_time"]
    while i<MAX_WAIT_TIME:#wait for message
        i+=1
        receiver = service_bus_client.get_queue_receiver(queue_name=queue_name, max_wait_time=30)
        with receiver:
            messages = receiver.receive_messages(max_wait_time=1)
            if messages:
                message_str=str(messages[0])
                #'completion_msg--container_name--float(phi_z)--config/run--int(n)--int(N)'
                if message_str.startswith("done"): 
                    receiver.complete_message(messages[0])
                    message_split=message_str.split("--")
                    return message_split[-1] # return N       
                else:
                    print(f"message kinda wierd ngl -_-: {message_str}")
                    receiver.complete_message(messages[0])
                    i=0
            else: 
                print(f"waiting for turn completion for {i} of {MAX_WAIT_TIME} sec.")
                time.sleep(config["sleep_time"])
    return 0

def log(service_bus_client,log,sender_info="DATA_PREP-"):
    try:
        sender = service_bus_client.get_queue_sender(queue_name="log")
        message=sender_info+str(datetime.now())+": "+log
        with sender:  
            sender.send_messages(ServiceBusMessage(message))
            print("log: %s"%(message))
    except exceptions.MessageSizeExceededError:
        print("queue full: %s"%(log))

#density calculation
def density(pcd,plane_model):
    plane_normal=plane_model[:3]
    points_pcd=np.asarray(pcd.points)#load points

    rotation=rotation_matrix(plane_normal)   #calc new coordinate system
    x_axis=norm(np.dot(rotation, [1,0,0]))
    y_axis=norm(np.cross(x_axis,plane_normal))
    points_2d=np.zeros((len(points_pcd),2))
    dot_x    = np.dot(points_pcd, x_axis)#put points in new coordinate system
    dot_y    = np.dot(points_pcd, y_axis)
    points_2d=np.c_[dot_x,dot_y]
    #get indexes of min and max on the x and y axis
    min_x=points_2d[np.argmin(dot_x,axis=0)]
    min_y=points_2d[np.argmin(dot_y,axis=0)]
    max_x=points_2d[np.argmax(dot_x,axis=0)]
    max_y=points_2d[np.argmax(dot_y,axis=0)]
    
    #find largest diagonal returns two corners opposite to another
    diagonal_index=np.argmax([np.linalg.norm(min_x-max_x),np.linalg.norm(min_y-max_y)])
    diagonal_vector=norm([min_x-max_x,min_y-max_y][diagonal_index])   
    diagonal_points=[np.asarray([min_x,max_x]),np.asarray([min_y,max_y])][diagonal_index]   
    orthogonal=norm([diagonal_vector[1],-diagonal_vector[0]])
    dot_orthogonal=np.dot(points_2d,orthogonal)
    
    min_orth=points_2d[np.argmin(dot_orthogonal)]
    max_orth=points_2d[np.argmax(dot_orthogonal)]

    L_1=max(np.linalg.norm(diagonal_points[0]-min_orth),np.linalg.norm(diagonal_points[0]-max_orth))
    L_s=min(np.linalg.norm(diagonal_points[0]-min_orth),np.linalg.norm(diagonal_points[0]-max_orth))
    L_2=max(np.linalg.norm(diagonal_points[1]-min_orth),np.linalg.norm(diagonal_points[1]-max_orth))
    
    phi=np.arccos(np.dot(diagonal_points[0]-min_orth,diagonal_points[0]-max_orth)/(np.linalg.norm(diagonal_points[0]-min_orth)*np.linalg.norm(diagonal_points[0]-max_orth)))
    area=(L_1+L_2)*L_s*np.sin(phi)/2 # trapezoid
    number_of_points=len(points_pcd)
    density=number_of_points/area
    return density

#############
"""EXECUTE"""
#############
#setup
current_dir=os.path.dirname(os.path.abspath(__file__))
config=load_config(current_dir)
part_name="part3_"

mode=config["mode"]
temp_dir=os.path.join(current_dir,"temp")
download=False
upload_bool=True

d_threshold=0.005

pipename_raspberry_pi_to_local= "ralm"
pipename_local_to_raspberry_pi= "lmra"
pipename_local_to_data_prep = "lmdp"
pipename_data_prep_to_local= "dplm"
pipename_data_prep_to_ID_NN= "dpin"
pipename_data_prep_to_ID_reg= "dpir"
pipename_img_gen_to_train= "imtr"
pipename_local_to_img_gen= "lmim"

######################################
# GET POINT CLOUD FROM SERVER AND 
# DOWNLOAD IT IF AVAILABLE DOWNLOAD 
######################################
servicebus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connection_string'])
queue_name=config["queue_name"]+pipename_local_to_data_prep
container_name_down=config["container_name"]+pipename_local_to_data_prep
container_client_down = ContainerClient.from_connection_string(config["azure_storage_connection_string"], container_name_down)

#moved to function download_files

########################################
#CONFIGURATION 
########################################
second_plane=True#turntable second biggest plane visible to the camera
config_name=os.path.join(current_dir,"config.pkl")
if (not os.path.exists(config_name)): #runs only if config.pkl does not exist
    print("--------------CONFIG FILE DOES NOT EXIST --> RUN CONFIGURATION----------------")
    pkl_name=os.path.join(temp_dir,"normals_dict.pkl")
    if not os.path.exists(pkl_name): #load planes N_1 and N_0
        #load files
        download_files(part_number=0,container_client=container_client_down,prefix="config")
        #find plane 
        normals_dict={
            "name":[],
            "N_0":[],
            "N_1":[],
            "center:":[],
            "density":[],
        }  

        for filename in os.listdir(temp_dir):
            if filename.startswith("config0"):
                print(filename)
 
                filepath=os.path.join(temp_dir,filename)
                pcd=o3d.io.read_point_cloud(filepath)

                #finds turntable RANSAC and DBSCAN and its center --> cut out cylinder
                if second_plane:
                    pcd,larger_plane,_ = RANSAC(pcd,d_threshold)
                _,turntable_pcd_plane,turntable_plane_model = RANSAC(pcd,d_threshold)
                seg, pcd_clean = DBSCAN(turntable_pcd_plane,d_threshold/5)
                center=pcd_clean.get_center()#estimate center
                N_z=norm(turntable_plane_model[:3])#normal of turntable plane model
                N_x=norm(np.cross(N_z,[1,0,0]))
                N_y=norm(np.cross(N_z,N_x))
                
                #cylinder removal
                inliers, inlier_cloud, outlier_cloud = cylinder_removal(pcd, center=center, N_z=N_z, radius=0.5, min_z=0.02)
                _, plane, plane_model = RANSAC(inlier_cloud,d_threshold/10)

                #density calculation
                _, plane_rough, _ = RANSAC(inlier_cloud,d_threshold/5)
                density_config=density(plane_rough,plane_model)#density in points/mm2 = f(pcd_plane,plane_model_exact)
                print("density: %s points/mm2"%(density_config))


                #center sphere
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.translate(center)
                sphere.compute_triangle_normals()

                #axis
                points=[
                    center,
                    center+N_z,
                    center+N_y,
                    center+N_x,
                ]

                lines=[
                    [0,1],
                    [0,2],
                    [0,3],
                ]
                line_set=o3d.geometry.LineSet()
                line_set.points= o3d.utility.Vector3dVector(points)
                line_set.lines= o3d.utility.Vector2iVector(lines)

                #plane model to dictionary 
                # print(normals_dict["name"])
                if len(normals_dict["name"])==0: 
                    normals_dict["name"]=[filename]
                    normals_dict["N_0"]=[turntable_plane_model]
                    normals_dict["N_1"]=[plane_model]
                    normals_dict["center"]=[center]
                    normals_dict["density"]=[density_config]
                else:
                    normals_dict["name"]=normals_dict["name"]+[filename]
                    normals_dict["N_0"]=(normals_dict["N_0"])+[turntable_plane_model]
                    normals_dict["N_1"]=(normals_dict["N_1"])+[plane_model]
                    normals_dict["center"]=normals_dict["center"]+[center]
                    normals_dict["density"]=normals_dict["density"]+[density_config]
        pickle.dump(normals_dict,open(pkl_name, "wb"))
        time.sleep(5)


    #load center, plane_models,  from pickle 
    normals_dict=pickle.load(open(pkl_name, "rb")) #unload pickle
    print(normals_dict)
    names=normals_dict["name"]
    densities=np.asarray(normals_dict["density"])
    turntable_plane_models=np.asarray(normals_dict["N_0"])
    plane_models=np.asarray(normals_dict["N_1"])
    centers=np.asarray(normals_dict["center"])
    turntable_plane_model=np.average(turntable_plane_models,axis=0)
    turntable_normal=norm(turntable_plane_model[:3])
    initial_center=np.average(centers,axis=0)
    rotation=rotation_matrix(turntable_normal)

    #abstraction to 2d
    line_models=[]
    # intersections=[] #comment out in final version
    # planes=[] #comment out in final version

    for plane_model in plane_models:
        line_model=line_equation(plane_model,turntable_plane_model)
        line_models.append(line_model)

        # #for visualization of points comment out in final version
        # plane_points=list_of_points_on_plane(plane_model, 1000)
        # plane = o3d.geometry.PointCloud()
        # plane.points = o3d.utility.Vector3dVector(plane_points)
        # plane.paint_uniform_color([0,1,0])
        # planes.append(plane)

    normal_index=find_normal_line(initial_center, line_models,rotation) # get config scan with perpendicular to camera
    density_config=densities[normal_index]/10e6
    print("density=%s points/mmm2"%(density_config))
    # #visualization of line intersection comment out in final version
    # i=0
    # for line_model in line_models:
    #     line_points=list_of_points_on_line(line_model,100)
    #     intersection=o3d.geometry.PointCloud()
    #     intersection.points = o3d.utility.Vector3dVector(line_points)
    #     if i==normal_index:intersection.paint_uniform_color([1,0,0])
    #     else:intersection.paint_uniform_color([.5,0,0])
    #     i+=1
    #     intersections.append(intersection)


    #############################################
    #minimize the objective function
    #the describes the difference in distance
    #from the planes to the center
    #############################################
    # spheres1, line_set1,_=create_spheres_and_lines(initial_center,line_models,[0.5,0,0.5],[1,0,1]) # comment out in final version
    difference=difference_between_distances(initial_center, (line_models,(turntable_plane_model,initial_center)))#initial difference calulated
    input=[line_models,(turntable_plane_model,initial_center)]
    optimize_object=minimize(difference_between_distances, initial_center, args=(input))
    center_init=optimize_object.x
    center = center_init-turntable_normal*distance_point_plane(center_init,turntable_plane_model)
    if distance_point_plane(center,turntable_plane_model)!=0:
        center = center_init+turntable_normal*distance_point_plane(center_init,turntable_plane_model)

    new_difference=difference_between_distances(center, (line_models,(turntable_plane_model,initial_center)))
    if not optimize_object.success: print(f"optimization failed with an final objective of {new_difference} terminated due:   {optimize_object.message}")
    print(f"Final objective: {new_difference}, before: {difference} - with an origin of: {center}")
    _, _, points_3d=create_spheres_and_lines(center,line_models,[0.5,0,0],[1,0,0])

    
    ###########################
    #points from the 3D scans 
    #are distorted in 
    #the circular shape is 
    #scanned as eclipse
    #this function fits to
    #said eclipse
    ###########################

    points_2d = points_on_plane_to_2D(np.asarray(points_3d),turntable_normal) #abstracted back to 2D

    coeffs = fit_ellipse(points_2d[:,0],points_2d[:,1]) # fit 2D eclipse
    x0, y0, ap, bp, e, phi = eclipse_param= cart_to_pol(coeffs) # get eclipse parameters
    x, y = get_ellipse_pts((x0, y0, ap, bp, e, phi)) #eclipse function
    # points_eclipse = np.c_[x, y,points_2d[1,2]*np.ones((len(x)))] # comment out in final version
    # points_eclipse = points_on_xy_to_3d(points_eclipse,turntable_normal) #cast to 3D plane # comment out in final version
    center_point_eclipse = [x0, y0,points_2d[1,2]] # get 2D center points
    center_point_eclipse = points_on_xy_to_3d([center_point_eclipse],turntable_normal)[0] # cast to 3D plane
    center_point_eclipse = center_point_eclipse+turntable_normal*distance_point_plane(center_point_eclipse,turntable_plane_model)
    print("found new centerpoint using eclipse: %s"%(center_point_eclipse))

    # eclipse_lines=[] # for visualization comment out in final version
    # i=0
    # for point_eclipse in points_eclipse:# comment out in final version
    #     if i==0:eclipse_lines.append([len(points_eclipse)-1,0])
    #     else: 
    #         if i==len(points_eclipse):eclipse_lines.append([i,0])
    #         else: eclipse_lines.append([i-1,i])
    #     i+=1
    # colors_eclipse_lines = [[1,0, 1] for i in range(len(eclipse_lines))]# comment out in final version
    # eclipse_lines_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(points_eclipse),
    #     lines=o3d.utility.Vector2iVector(eclipse_lines),
    # )
    # eclipse_lines_set.colors = o3d.utility.Vector3dVector(colors_eclipse_lines)# comment out in final version
    # sphere_center_eclipse = o3d.geometry.TriangleMesh.create_sphere(radius=0.01) # comment out in final version
    # sphere_center_eclipse.compute_vertex_normals()
    # sphere_center_eclipse.paint_uniform_color([1,0,1])
    # sphere_center_eclipse.translate(center_point_eclipse)
    # spheres2.append(sphere_center_eclipse)
    ################################
    #distortion removal by fitting
    #the previously calculated 
    #eclipse to a circle 
    ################################

    circle_param=x0, y0, min(ap,bp) # use smallest radius on eclipse to create circle
    # x_circle, y_circle = get_circle_pts(circle_param) #comment out in final
    # points_circle = np.c_[x_circle, y_circle,points_2d[1,2]*np.ones((len(x)))] #comment out in final
    # points_circle = points_on_xy_to_3d(points_circle,turntable_normal) #comment out in final
    # circle_lines=[] #comment out in final
    # i=0
    # for point_circle in points_circle:#comment out in final
    #     if i==0:circle_lines.append([len(points_circle)-1,0])#comment out in final
    #     else: 
    #         if i==len(points_circle):circle_lines.append([i,0])#comment out in final
    #         else: circle_lines.append([i-1,i])#comment out in final
    #     i+=1
    # colors_circle_lines = [[0.8,0.8, 0.8] for i in range(len(circle_lines))]#comment out in final
    # circle_lines_set = o3d.geometry.LineSet(#comment out in final
    #     points=o3d.utility.Vector3dVector(points_circle),#comment out in final
    #     lines=o3d.utility.Vector2iVector(circle_lines),#comment out in final
    # )
    # circle_lines_set.colors = o3d.utility.Vector3dVector(colors_circle_lines)#comment out in final

    # t=time.time()
    # points_eclipse_new=correction_ellipse(points_eclipse,turntable_normal,center_point_eclipse,circle_param,eclipse_param)#correct 
    # print(f"correction for {len(points_eclipse_new)} points took: {time.time()-t} seconds")

    # new_eclipse_lines=[]
    # i=0
    # for coord_eclipse_new in points_eclipse_new: #comment out in final
    #     if i==0:new_eclipse_lines.append([len(points_eclipse_new)-1,0])#comment out in final
    #     else: 
    #         if i==len(points_eclipse_new):new_eclipse_lines.append([i,0])#comment out in final
    #         else: new_eclipse_lines.append([i-1,i])
    #     i+=1
    # colors_new_eclipse_lines = [[1,0, 0] for i in range(len(new_eclipse_lines))]#comment out in final
    # new_eclipse_lines_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(points_eclipse_new),#comment out in final
    #     lines=o3d.utility.Vector2iVector(new_eclipse_lines),#comment out in final
    # )
    # new_eclipse_lines_set.colors = o3d.utility.Vector3dVector(colors_new_eclipse_lines)#comment out in final

    #save config
    config_dict={ #structure in dictionary
        "turntable_normal":turntable_normal,
        "center":center_point_eclipse,
        "circle_param":circle_param,
        "eclipse_param":eclipse_param,
        "plane_models": plane_models,
        "normal_index":normal_index,
        "density": density_config,
    }

    #pickle dictionary
    pickle.dump(config_dict,open(config_name, "wb"))

#load config from pickle
config_dict=pickle.load(open(config_name, "rb")) #unload pickle
print(config_dict)
turntable_normal=config_dict["turntable_normal"]
center_turntable=config_dict["center"]
circle_param=config_dict["circle_param"]
eclipse_param=config_dict["eclipse_param"]
plane_models=config_dict["plane_models"]
normal_index=config_dict["normal_index"]
density_config=config_dict["density"]

#########################################
#LOCAL REGISTRATION
#########################################

PARTS=49
time_total=time.time()
for part_number in range(PARTS):
    # part_number=3
    time_prep=time.time()
    #load from cloud 
    download_files(part_number=part_number, container_client=container_client_down)
    #load point clouds
    part_name="%s%s_scan"%("part",part_number)
    pcds=[]
    file_names=[]
    for file in os.listdir(temp_dir):
        if str(file).startswith(part_name):
            pcds.append(o3d.io.read_point_cloud(os.path.join(temp_dir,file)))
            file_names.append(file)
            print("loaded the file %s from storage"%(file))
            os.remove(os.path.join(temp_dir,file))

    #distortion removal and clean up 
    pcds_new=[]
    t=time.time()
    for pcd in pcds:
        points_new=correction_ellipse(np.asarray(pcd.points),turntable_normal,center_turntable,circle_param,eclipse_param) # correct for ellipse
        points_new=correction_y(points_new, turntable_normal,plane_models[normal_index],center_turntable) # correct for distortion around x-achis
        pcd.points=o3d.utility.Vector3dVector(points_new) #load in correct points into old point cloud
        _, pcd_new, _ = cylinder_removal(pcd,center=center_turntable,N_z=turntable_normal,radius=0.3,min_z=0.009, max_z=0.2) #remove point outside of defined cylinder
        _, pcd_new=DBSCAN(pcd_new,d_threshold) # removed outliers that are not connected to point cloud
        pcds_new.append(pcd_new) #add to new pcd list
    print("it took %s seconds for cylinder and distortion removal and DBSCAN for %s point clouds"%(time.time()-t,len(pcds)))
    steps=len(pcds_new)
    degree=(2*np.pi/len(pcds))

    #turn around angle and local registration pairwise
    pcd_comb=o3d.geometry.PointCloud()
    pcd_prev=o3d.geometry.PointCloud()
    for step in range(len(pcds_new)):
        pcd_cur=pcds_new[step]
        if step==0: 
            pcd_prev=copy.deepcopy(pcd_cur)
            pcd_comb=copy.deepcopy(pcd_cur)
            continue

        #TURN
        transform_mat=transform_around_axis(degree*step, turntable_normal,center_turntable, rotation_matrix(turntable_normal))
        pcd_cur.transform(transform_mat)

        #ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_cur, pcd_prev, 0.03, np.eye(4), # source, target, max_correspondence_distance
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))
        
        pcd_cur.transform(reg_p2p.transformation)
        
        print("registration results for pcd %s of %s:  %s"%(step,steps,reg_p2p))
        
        if reg_p2p.fitness>0.1 and reg_p2p.inlier_rmse<0.004:
            pcd_prev=pcd_cur
            pcd_comb=pcd_comb+pcd_cur

    _, pcd_comb=DBSCAN(pcd_comb,d_threshold/5) # clean up scan artifacts
    log(servicebus_client, "data prep for part %s took %s seconds"%(part_number,time.time()-time_prep))
    #####################
    # SEND TO STORAGE 
    # CONTAINER AND
    # SEND MESSAGE TO 
    # IDENTIFICATION 
    # CONTAINER
    #####################

    upload_path=os.path.join(current_dir,"upload")
    container_name_upload=config["container_name"]+pipename_data_prep_to_ID_reg
    queue_name_upload=config["queue_name"]+pipename_data_prep_to_ID_reg
    if mode=="NN":
        container_name_upload=config["container_name"]+pipename_data_prep_to_ID_NN
        queue_name_upload=config["queue_name"]+pipename_data_prep_to_ID_NN

    #save locally
    upload_file_name=file_names[0].split("_")[0]+".pcd"
    upload_file_path=os.path.join(upload_path,upload_file_name)
    print("saving locally to: %s"%(upload_file_name))
    # o3d.visualization.draw_geometries([pcd_comb])
    o3d.io.write_point_cloud(upload_file_path,pcd_comb)

    # upload to storage
    if upload_bool:
        container_client = ContainerClient.from_connection_string(config["azure_storage_connection_string"], container_name_upload)   
        if not container_client.exists():
            print(f"creating new container with {container_name_upload}")
            try:
                container_client.create_container()
            except exceptions.ResourceExistsError:
                print("waiting for deletion")
                time.sleep(20)
                container_client.create_container()
        blob_client = container_client.get_blob_client(upload_file_name)
        with open(upload_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
            print(f"{upload_file_name} uploaded")
        os.remove(upload_file_path)

    servicebus_client = ServiceBusClient.from_connection_string(conn_str=config['azure_messaging_connection_string'])
    sender = servicebus_client.get_queue_sender(queue_name=queue_name_upload)
    with sender:  
            #'done--{container_name}--{part0.pcd}'
        message=f"done--{container_name_upload}--{upload_file_name}"   
        sender.send_messages(ServiceBusMessage(message))
        print(f"sent message: {message}")

    # sphere_center_eclipse, line_set=coordinate_system(turntable_normal,center_turntable)
    # o3d.visualization.draw_geometries([sphere_center_eclipse,line_set,pcd_comb]) #remove all visualization in final version
log(servicebus_client, "Took a total time of %s seconds"%(time.time()-time_total))

