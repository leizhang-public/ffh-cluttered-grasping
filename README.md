<h1 align="center">ContactDexNet: Multi-fingered Robotic Hand Grasping in Cluttered Environments through Hand-object Contact Semantic Mapping</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2404.08844">
        <img src="https://img.shields.io/badge/arXiv-2404.08844-B31B1B.svg?style=plastic&logo=arxiv" alt="arXiv">
    </a>
    <a href="https://sites.google.com/view/contact-dexnet">
        <img src="https://img.shields.io/badge/Website-ContactDexNet-purple?style=plastic&logo=Google%20chrome" alt="Website">
    </a>
</p>
<p align="center">
    <a href="https://leizhang-public.github.io/">Lei Zhang</a>, 
    <a href="https://baikaixin-public.github.io/">Kaixin Bai</a>, 
    <a href="">Guowen Huang</a>, 
    <a href="https://www.ce.cit.tum.de/air/people/zhenshan-bing-drrernat/">Zhenshan Bing</a>, 
    <a href="https://ieeexplore.ieee.org/author/37404312400">Zhaopeng Chen</a>, 
    <a href="https://www.ce.cit.tum.de/air/people/prof-dr-ing-habil-alois-knoll/">Alois Knoll</a>, 
    <a href="https://ieeexplore.ieee.org/author/37281460600">Jianwei Zhang </a>
</p>
<p align="center">University of Hamburg, Agile Robots, Technical University of Munich</p>
<p align="center">
    <a href="https://sites.google.com/view/contact-dexnet" target="_blank">
        <img src="./images/fig_arnie_gtc.jpeg" alt="ContactDexNet" width="40%" height="40%" border="0" />
    </a>
    <!-- <a href="https://sites.google.com/view/contact-dexnet" target="_blank">
        <img src="./images/fig_head.png" alt="EAgent" width="40%" height="40%" border="0" />
    </a> -->
</p>

## Example of ContactDexNet dataset (previous FFHClutteredGrasping dataset)
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

# BibTex

If you find our work helpful, please consider citing it:

```bash
@article{zhang2024multi,
  title={Multi-fingered robotic hand grasping in cluttered environments through hand-object contact semantic mapping},
  author={Zhang, Lei and Bai, Kaixin and Huang, Guowen and Bing, Zhenshan and Chen, Zhaopeng and Knoll, Alois and Zhang, Jianwei},
  journal={arXiv preprint arXiv:2404.08844},
  year={2024}
}

@misc{zhang2025contactdexnetmultifingeredrobotichand,
      title={ContactDexNet: Multi-fingered Robotic Hand Grasping in Cluttered Environments through Hand-object Contact Semantic Mapping}, 
      author={Lei Zhang and Kaixin Bai and Guowen Huang and Zhenshan Bing and Zhaopeng Chen and Alois Knoll and Jianwei Zhang},
      year={2025},
      eprint={2404.08844},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2404.08844}, 
}
```