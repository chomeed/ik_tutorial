This repo is made to demonstrate Jacobian Pseudoinverse-based IK solver on MuJoCo environment. Additionally, it applies joint limit for the 3D robots.

### Demo 
[2D](videos/2d_ik_demo.mov) \
[3D](videos/3d_ik_demo.mov) \
[Z1](videos/z1_ik_demo.mov)

### Dependency

    mujoco
    numpy

### How to Use 

2D RRR Robot: 

    python mujoco_interface_2d.py 


Basic 3D RRRRRR Robot: 

    python mujoco_interface_3d.py

    
Unitree Z1 Robot (without gripper): 

    python mujoco_interface_z1.py


### Keys 

#### Translation
Up, Down arrow : (x-axis) \
Left, Right arrow : (y-axis) \
O, L :  (z-axis)

#### Rotation
Q, W : (x-axis) \
A, S : (y-axis) \
Z, X : (z-axis) 
