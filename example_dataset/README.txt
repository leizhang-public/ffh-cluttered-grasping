The structure of scene dataset:
scene_dataset
 - scene0.obj, the scene mesh file
 - scene0.pt, is a dictionary, contains all information of the scene. It contains keys:
    - "object_random_translations", a dictionary {object_name: the random offset 3x1 assigned on the object when constructing the scene}
    - "object_random_transformations_all", a dictionary {object_name : the random transformation 4x4 matrix assigned on the object when constructing the scene}
    - "object_pointcloud", the pointcloud file of the scene
    - "camera_extrinsic", the extrinsic matrix of the camera 
    - "camera_intrinsic", the intrinsic matrix of the camera 
    - "camera_resolution", the resolution of the camera 
    - "object_camera_pointcloud", the scene pointcloud under the camera view
    - "original_hand_pose", a dictionary {object_name: [original hand pose of the grasp]}
    - "grasps", a dictionary {object_name: [(normalized hand pose of the grasp, the object scale related to the grasp)]}
    - "contacts", a dictionary {object_name: [(segmentation map, contact map, filtered contact map)]}
    - "grasp_scores", a dictionary {object_name: [validated grasp score of the grasp]}
    - "radius_pointcloud", a dictionary {object_name: [sampled pointcloud around the grasp]}
    - "collision", a dictionary {object_name: [whether the grasp collides with other objects in the scene]}
    - "centroid", a dictionary {object_name: [the centroid of the grasp]}

The structure of affordance dataset:
affordance_dataset
 - affordance_{object_name}.npy, the affordance information of the object
 - {object_name}.ply, the pointcloud that can be used to visualize the affordance information 

The visualization python scripts:
 - visualize_affordance.py, visualize the affordance information given the object name 
 - visualize_scene.py, visualize the scene and information contained given the pt file name
