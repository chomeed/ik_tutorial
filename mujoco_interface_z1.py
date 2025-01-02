from mujoco_interface_3d import Interface
import numpy as np

z1_interface = Interface("mjcf/z1_robot.xml", root_name="link01")
z1_interface.robot.chain.add_eef_offset(np.array([0.051, 0, 0]))
z1_interface.simulate()