import os
import copy

from tqdm import tqdm
import numpy as np
import open3d as o3d
import torch
import trimesh as tm

def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)   
    return out

def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v

def robust_compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]

    x = normalize_vector(x_raw) 
    y = normalize_vector(y_raw) 
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)
    return matrix

def update_hand_poses(hand_poses):
    translation = hand_pose[:, 0:3]
    rotation = robust_compute_rotation_matrix_from_ortho6d(hand_pose[:, 3:9])
    return translation, rotation

def get_color(collision, score):
    green = (111 / 255, 158 / 255, 105 / 255)
    red = (184 / 255, 94 / 255, 90 / 255)
    if collision:
        return (85 / 255, 147 / 255, 186 / 255)
    elif score >= 0.5:
        return green
    else: return red

def get_finger_color(value):
    if value > 1:
        value = (value - 2) // 4 + 2
    colors = [(153/255, 204/255, 1), (153/255, 204/255, 1),
    (0.0, 0.996078431372549, 0.20784313725490197), 
    (0.41568627450980394, 0.4627450980392157, 0.9882352941176471), 
    (0.996078431372549, 0.8313725490196079, 0.7686274509803922), 
    (0.996078431372549, 0.0, 0.807843137254902), 
    (0.050980392156862744, 0.9764705882352941, 1.0), 
    (0.9647058823529412, 0.9764705882352941, 0.14901960784313725), 
    (1.0, 0.5882352941176471, 0.08627450980392157), 
    (0.2784313725490196, 0.6078431372549019, 0.3333333333333333), 
    (0.9333333333333333, 0.6509803921568628, 0.984313725490196), 
    (0.8627450980392157, 0.34509803921568627, 0.49019607843137253), 
    (0.8392156862745098, 0.14901960784313725, 1.0), 
    (0.43137254901960786, 0.5372549019607843, 0.611764705882353), 
    (0.0, 0.7098039215686275, 0.9686274509803922), (0.7137254901960784, 0.5568627450980392, 0.0), 
    (0.788235294117647, 0.984313725490196, 0.8980392156862745), (1.0, 0.0, 0.5725490196078431), 
    (0.13333333333333333, 1.0, 0.6549019607843137), 
    (0.8901960784313725, 0.9333333333333333, 0.6196078431372549), 
    (0.5254901960784314, 0.807843137254902, 0.0), (0.7372549019607844, 0.44313725490196076, 0.5882352941176471), 
    (0.49411764705882355, 0.49019607843137253, 0.803921568627451), (0.9882352941176471, 0.4117647058823529, 0.3333333333333333), 
    (0.8941176470588236, 0.5607843137254902, 0.4470588235294118), (153/255, 204/255, 1),]
    return colors[value]

def get_joint_color(value):
    colors = [(153/255, 204/255, 1), (153/255, 204/255, 1),
    (0.0, 0.996078431372549, 0.20784313725490197), 
    (0.41568627450980394, 0.4627450980392157, 0.9882352941176471), 
    (0.996078431372549, 0.8313725490196079, 0.7686274509803922), 
    (0.996078431372549, 0.0, 0.807843137254902), 
    (0.050980392156862744, 0.9764705882352941, 1.0), 
    (0.9647058823529412, 0.9764705882352941, 0.14901960784313725), 
    (1.0, 0.5882352941176471, 0.08627450980392157), 
    (0.2784313725490196, 0.6078431372549019, 0.3333333333333333), 
    (0.9333333333333333, 0.6509803921568628, 0.984313725490196), 
    (0.8627450980392157, 0.34509803921568627, 0.49019607843137253), 
    (0.8392156862745098, 0.14901960784313725, 1.0), 
    (0.43137254901960786, 0.5372549019607843, 0.611764705882353), 
    (0.0, 0.7098039215686275, 0.9686274509803922), (0.7137254901960784, 0.5568627450980392, 0.0), 
    (0.788235294117647, 0.984313725490196, 0.8980392156862745), (1.0, 0.0, 0.5725490196078431), 
    (0.13333333333333333, 1.0, 0.6549019607843137), 
    (0.8901960784313725, 0.9333333333333333, 0.6196078431372549), 
    (0.5254901960784314, 0.807843137254902, 0.0), (0.7372549019607844, 0.44313725490196076, 0.5882352941176471), 
    (0.49411764705882355, 0.49019607843137253, 0.803921568627451), (0.9882352941176471, 0.4117647058823529, 0.3333333333333333), 
    (0.8941176470588236, 0.5607843137254902, 0.4470588235294118), (153/255, 204/255, 1)]
    return colors[value]

def get_gradient_color(ratio):
    color = [153/255, 204/255, 1]
    scale = (0.5 + 0.5*( 1 - ratio), 0.7 + 0.3*( 1 - ratio), 0.8 + 0.2*( 1 - ratio))
    return (color[0] * scale[0], color[1] * scale[1], color[2] * scale[2])

def get_distance_map_color(ratio):
    color = [153/255, 204/255, 1]
    scale = (0.2 + 0.8*( 1 - ratio), 0.2 + 0.8*( 1 - ratio), 0.2 + 0.8*( 1 - ratio))
    return (color[0] * scale[0], color[1] * scale[1], color[2] * scale[2])

if __name__ == '__main__':
    dataset_directory = 'scene_dataset'
    paths = os.listdir(dataset_directory)
    color_values = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 2, 3, 4, 5]

    for name in paths:
        name = name.split('.')[0]
        visualize_contents = []
        pcds = []

        pt_file_path = os.path.join(dataset_directory, f"{name}.pt")
        model_data_dir = os.path.join(dataset_directory, f"{name}.obj")
        pt_data = torch.load(pt_file_path, map_location="cpu")
        object_names = list(pt_data['grasps'].keys())

        manager = tm.collision.CollisionManager()
        selected_hands = []
        visualize_hands = []
        for obj_name in tqdm(object_names):
            length = len(pt_data['centroid'][obj_name])
            for index in tqdm(range(0, length)):
                hand_pose = pt_data['grasps'][obj_name][index][0]
                centroid = pt_data['centroid'][obj_name][index]
                object_pointcloud = pt_data['object_pointcloud']

                hand_pose[:, :3] = hand_pose[:, :3] + centroid
        
                translation, rotation = update_hand_poses(hand_pose)
                rotation, translation = rotation[0].cpu().numpy(), translation.cpu().numpy()
                translation = translation.T
                # rotation = np.linalg.inv(rotation)
                transformation = np.concatenate((rotation, translation), axis=1)
                transformation = np.concatenate((transformation, np.array([[0, 0, 0, 1]])), axis=0)
        
                # combined_hand = tm.creation.annulus(r_min=0.0, r_max=0.05, height=0.15, resolution=20, split=4)
                combined_hand = tm.load('palm_vis_simple.obj')
                combined_hand.apply_transform(transformation)
                is_collision = manager.in_collision_single(combined_hand)
                if not is_collision:
                    manager.add_object(obj_name + str(index), combined_hand)
                    selected_hands.append((obj_name, index))
                    collision = pt_data['collision'][obj_name][index]
                    score = pt_data['grasp_scores'][obj_name][index]
                    visualize_hands.append(combined_hand.as_open3d.compute_vertex_normals().paint_uniform_color(get_color(collision, score)))

        filtered_hands = {}
        filtered_visualize_hands = []
        cur_index = 0
        for obj_name, index in selected_hands:
            if obj_name not in filtered_hands.keys():
                filtered_hands[obj_name] = index
                filtered_visualize_hands.append(visualize_hands[cur_index])
            cur_index += 1

        distance_colorized_pcds = []
        joint_colorized_pcds = []
        finger_colorized_pcds = []

        for obj_name, index in filtered_hands.items():
            hand_pose = pt_data['grasps'][obj_name][index][0]
            centroid = pt_data['centroid'][obj_name][index]
            hand_pose[:, :3] = hand_pose[:, :3] + centroid
        
            translation, rotation = update_hand_poses(hand_pose)
            rotation, translation = rotation[0].cpu().numpy(), translation.cpu().numpy()
            translation = translation.T
            transformation = np.concatenate((rotation, translation), axis=1)
            transformation = np.concatenate((transformation, np.array([[0, 0, 0, 1]])), axis=0)

            segmentation_map, contact_map = pt_data['contacts'][obj_name][index][:2]
            contact_map = torch.squeeze(contact_map, dim=0)
            color_scale = contact_map[:, 3]
            colors = [get_distance_map_color(scale.item()) for scale in color_scale]

            distance_color = copy.deepcopy(colors)
            joint_color = copy.deepcopy(colors)
            finger_color = copy.deepcopy(colors)

            for i in range(0, segmentation_map.shape[0]):
                if segmentation_map[i].item() <= 1:
                    joint_color[i] = get_gradient_color((contact_map[:, :3][i][2].cpu().detach().item() + 0.08)/0.16)
                    finger_color[i] = get_gradient_color((contact_map[:, :3][i][2].cpu().detach().item() + 0.08)/0.16)
                else: 
                    joint_color[i] = get_joint_color(color_values[segmentation_map[i].item() - 1])
                    finger_color[i] = get_finger_color(color_values[segmentation_map[i].item() - 1])

            distance_pcd = o3d.geometry.PointCloud()
            joint_pcd = o3d.geometry.PointCloud()
            finger_pcd = o3d.geometry.PointCloud()
            
            distance_pcd.points = o3d.utility.Vector3dVector(torch.squeeze(contact_map[:, :3]).cpu().detach().numpy())
            joint_pcd.points = o3d.utility.Vector3dVector(torch.squeeze(contact_map[:, :3]).cpu().detach().numpy())
            finger_pcd.points = o3d.utility.Vector3dVector(torch.squeeze(contact_map[:, :3]).cpu().detach().numpy())

            random_transformation, object_scale = pt_data['object_random_transformations_all'][obj_name]
            distance_pcd.transform(random_transformation)
            joint_pcd.transform(random_transformation)
            finger_pcd.transform(random_transformation)

            distance_pcd.colors = o3d.utility.Vector3dVector(np.array(distance_color))
            joint_pcd.colors = o3d.utility.Vector3dVector(np.array(joint_color))
            finger_pcd.colors = o3d.utility.Vector3dVector(np.array(finger_color))

            distance_colorized_pcds.append(distance_pcd)
            joint_colorized_pcds.append(joint_pcd)
            finger_colorized_pcds.append(finger_pcd)
        
        visualize_contents.clear()
        visualize_contents.extend(distance_colorized_pcds)
        visualize_contents.extend(filtered_visualize_hands)
        o3d.visualization.draw_geometries(visualize_contents)

        visualize_contents.clear()
        visualize_contents.extend(joint_colorized_pcds)
        visualize_contents.extend(filtered_visualize_hands)
        o3d.visualization.draw_geometries(visualize_contents)

        visualize_contents.clear()
        visualize_contents.extend(finger_colorized_pcds)
        visualize_contents.extend(filtered_visualize_hands)
        o3d.visualization.draw_geometries(visualize_contents)