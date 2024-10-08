# Multi-fingered Robotic Hand Grasping in Cluttered Environments through Hand-object Contact Semantic Mapping

[Project Page](https://sites.google.com/view/ffh-cluttered-grasping) [arXiv](https://arxiv.org/abs/2404.08844v2) [PDF](https://arxiv.org/pdf/2404.08844v2)



## Example of FFHClutteredGrasping dataset
Grasping Candidates and Contact Distance Map.               |  Cluttered Scene
:-------------------------:|:-------------------------:
![Grasping Candidates](./images/example_dataset/scene_grasp_quality_object_file_eight_scene7_multiple_objects_distance.gif)![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png)Blue: Candidate under Collision, <br />![#52900C](https://placehold.co/15x15/52900C/52900C.png)Green: Positive Candidate, <br />![#C3352B](https://placehold.co/15x15/C3352B/C3352B.png) Red: Negative Candidate  |  ![Grasping Candidates](./images/example_dataset/scene_grasp_quality_object_file_eight_scene7_multiple_objects_pcd.gif)

Contact Semantic Map             |  Affordance Map
:-------------------------:|:-------------------------:
![Contact Semantic Map](./images/example_dataset/scene_grasp_quality_object_file_eight_scene7_multiple_objects_finger_no_hand.gif)  |  ![Affordance Map](./images/example_dataset/scene_grasp_quality_object_file_eight_scene7_multiple_objects_affordance.gif)
<!-- 
![Grasping Candidates](./images/example_dataset/scene_grasp_quality_object_file_eight_scene7_multiple_objects_distance.gif)
![Grasping Candidates](./images/example_dataset/scene_grasp_quality_object_file_eight_scene7_multiple_objects_pcd.gif) -->


## Visualize example dataset

visualize example scene with cluttered grasping data of multi-fingered robotic hand with following modalities:
- collision score
- grasping quality
- contact distance map
- contact semantic map
- affordance information

```python
cd example_dataset
# visualize cluttered scene with grasp candidates, and corresponding collision score, grasp qualities, contact distance and semantic information
python visualize_scene.py

# visualize the affordance information
python visualize_affordance.py

```
P.S: In visualization, the model of robotic hand is a simplied version.