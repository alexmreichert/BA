import open3d as o3d
import os
import json


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
        o3d.visualization.draw_geometries([pcd])
        print("finished scan")
    except RuntimeError:
        print("no cam connected")
