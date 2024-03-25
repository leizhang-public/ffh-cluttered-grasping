import os
import json

import open3d as o3d
import numpy as np
import pickle
import sys
sys.path.insert(0, os.getcwd())


if __name__ == '__main__':
    object_name = 'normal_bottle_ModelNet_bottle_0027'
    folder_base = 'affordance_dataset'
    file_name = 'object_point_cloud_' + object_name
    ply_file_name = "{}.ply".format(file_name)

    pcd = o3d.io.read_point_cloud(os.path.join(folder_base, ply_file_name))
    default_color = np.array([0.5,0.5,0.5]).reshape(1,3)
    size_pcd = np.asarray(pcd.points).shape
    npy_colors = np.repeat(default_color, size_pcd[0], 0)
    pcd.colors = o3d.utility.Vector3dVector(npy_colors)

    affordance_list = ["HandleGrasp", "WrapGrasp", "Press", "Pour", "Cut", "Stab", "Pull", "Push", "Open", "Twist", "Hammer", "Pry"]
    colors = np.array([(99, 110, 250), (239, 85, 59), (0, 204, 150), (171, 99, 250),(255, 161, 90), (27, 211, 243), (253, 101, 145), (182, 232, 128), (255, 151, 255), (252, 201, 80), (255, 255, 51), (0, 255, 255), (255, 255, 178)])

    content = {}
    with open(os.path.join(folder_base, object_name + ".pkl"), "rb") as f:
        content = pickle.load(f)
    for index_affordance, affordance_key in enumerate(affordance_list):
        if affordance_key in content.keys():
            mask_list = content[affordance_key]
            pcdvis = o3d.geometry.PointCloud()
            pcdvis.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
            pcdvis.paint_uniform_color(colors[0] / 255)
            pcdvis2 = o3d.geometry.PointCloud()
            pcdvis2.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask_list])
            pcdvis2.paint_uniform_color(colors[index_affordance + 1] / 255)
            o3d.visualization.draw_geometries([pcdvis, pcdvis2])
              