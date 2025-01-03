import mujoco as mj
from mujoco.glfw import glfw
import numpy as np 
from kinematics_3d import * 
from transform_utils import *


class RobotController: 
    def __init__(self, xml_path, root_name="robot"): 
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        print("jnts", self.model.jnt_axis)
        self.target_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "target")
        print(self.target_id)
        # print(self.data.xpos) 

        # self.model.opt.timestep = 0.002 # 500 Hz 
        # self.model.opt.timestep = 1 # 5 Hz

        print(f"nq: {self.model.nq} \nnv: {self.model.nv}")
        print(f"qpos: {self.data.qpos}", type(self.data.qpos))

        
        robot_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, root_name)
        jnt_start = self.model.body_jntadr[robot_body_id]

        # dof = self.model.body_jntnum[robot_body_id]
        dof = 6 # NOTE: hardcoded for now
        print("robot_body_id", robot_body_id)
        print("dof", dof)

        mj.mj_step(self.model, self.data) # Need to call mj_step to update xpos and xmat
        
        joints = [] 
        # xyz_parent = np.zeros(3)
        # rpy_parent = np.eye(3) # when using rotation matrix representation
        rpy_parent = np.zeros(3) # Euler angles representation
        for jnt_id in range(jnt_start, jnt_start + dof): 
            joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
            joint_type = self.model.jnt_type[jnt_id]
            
            # Axis of rotation(Joint) 
            axis = self.model.jnt_axis[jnt_id]

            # Initial XYZ of the body(Frame) w.r.t the world frame 
            # xyz = self.data.xpos[self.model.jnt_bodyid[jnt_id]] # global frame 
            xyz_relative = self.model.body_pos[self.model.jnt_bodyid[jnt_id]]


            # Initial RPY of the body(Frame) w.r.t the world frame 
            rpy = self.data.xmat[self.model.jnt_bodyid[jnt_id]]
            rpy = np.reshape(rpy, (3, 3))
            rpy = rpy_from_rot(rpy)

            # Compute position and orientation w.r.t the parent body 
            # xyz_relative = xyz - xyz_parent # global frame 
            # rpy = rpy_parent.T @ rpy # when using rotation matrix representation
            rpy_relative = rpy - rpy_parent

            print(f"Joint body ID: {self.model.jnt_bodyid[jnt_id]}")
            print(f"  Joint: {joint_name} (ID: {jnt_id}, Type: {joint_type}, Axis: {axis})")
            print(f"    XYZ(relative): {xyz_relative}")  
            # print(f"    XYZ(global): {xyz}")
            print(f"    RPY(relative): {rpy_relative}")
            joints.append(Joint(axis, xyz_relative, rpy_relative))

            # xyz_parent = xyz # if using global frame  
            rpy_parent = rpy

        frames = [Frame(joint) for joint in joints]
        self.chain = KinematicChain(frames)
        for frame in self.chain.frames: 
            print(frame)
        print("Initial EEF pose", self.chain.forward_kinematics(self.current_thetas)[-1][:3, 3])
        print("FK", self.chain.forward_kinematics(self.current_thetas))

    @property
    def current_thetas(self): 
        '''
        return data.qpos
        '''
        return self.data.qpos[:6]
    
    def get_desired_thetas(self):
        '''
        Return thetas based on the target position 
        '''
        target_pos = self.data.xpos[self.target_id] # x, y, z

        # NOTE: add orientation as well 
        target_ori = self.data.xmat[self.target_id].reshape(3, 3)        

        # IK solve
        new_thetas, info = self.chain.compute_ik_newton_rapshon(self.current_thetas, target_pos, target_ori)
        # print(f"Target position: {target_pos} | New thetas: {new_thetas}")
        return new_thetas, info

    def set_thetas(self, thetas): 
        '''
        set data.qpos = thetas
        '''
        self.data.qpos[:6] = thetas 

    def move_target(self, delta_pos, mode="translate"):
        '''
        Move the target ball by delta_pos(delta_x, delta_y)
        '''
        if mode == "translate":
            slide_x_idx = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "slide_x")
            slide_y_idx = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "slide_y")
            slide_z_idx = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "slide_z")

            self.data.qpos[slide_x_idx] += delta_pos[0]
            self.data.qpos[slide_y_idx] += delta_pos[1]
            self.data.qpos[slide_z_idx] += delta_pos[2]
        elif mode == "rotate":
            hinge_x_idx = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "hinge_x")
            hinge_y_idx = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "hinge_y")
            hinge_z_idx = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "hinge_z")

            self.data.qpos[hinge_x_idx] += delta_pos[0]
            self.data.qpos[hinge_y_idx] += delta_pos[1]
            self.data.qpos[hinge_z_idx] += delta_pos[2]


class Interface: 
    def __init__(self, xml_path, root_name="robot"): 
        self.robot = RobotController(xml_path, root_name)

        # Initial setup 
        self._setup_cam()
        self._setup_opt() 
        self._setup_glfw()     
        self._setup_scene_and_context() 
        self._setup_viewport()

        # Track Mouse Position 
        # glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)

        # Callback for Keyboard Input
        glfw.set_key_callback(self.window, self.key_callback)

        self.info = None 

    def _setup_cam(self): 
        self.cam = mj.MjvCamera()
        mj.mjv_defaultCamera(self.cam)

        self.cam.type = mj.mjtCamera.mjCAMERA_FIXED.value
        self.cam.fixedcamid = 0

    def _setup_opt(self): 
        self.opt = mj.MjvOption()
        mj.mjv_defaultOption(self.opt)

    def _setup_scene_and_context(self): 
        self.scene = mj.MjvScene(self.robot.model, maxgeom=100)
        self.context = mj.MjrContext(
            self.robot.model, mj.mjtFontScale.mjFONTSCALE_150.value)

    def _setup_glfw(self): 
        glfw.init()
        self.window = glfw.create_window(720, 480, "Robot Simulation", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

    def _setup_viewport(self):
        viewport_width, viewport_height = glfw.get_framebuffer_size(
            self.window)
        self.viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    def step(self, action=np.array([0, 0])):
        '''
        Args: 
            - action: np.ndarray of joint angles i.e., Shape (2,) 
        '''


        desired_thetas, info = self.robot.get_desired_thetas()
        self.robot.set_thetas(desired_thetas)
        mj.mj_step(self.robot.model, self.robot.data)
        mj.mjv_updateScene(self.robot.model, self.robot.data, self.opt, None, self.cam,
                            mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        
        mj.mjr_render(self.viewport, self.scene, self.context)

        

        info.update({"Current thetas": self.robot.current_thetas})
        info.update({"Control": "Use arrow keys to move the target ball"})

        np.set_printoptions(precision=3)
        for idx, (k, v) in enumerate(info.items()):
            mj.mjr_text(mj.mjtFont.mjFONT_NORMAL, f"{k}: {v}", self.context, 0.04, 0.2 - 0.025*idx, 255, 255, 255)

        mj.mjr_overlay(
            mj.mjtFont.mjFONT_NORMAL,  # Font type
            mj.mjtGridPos.mjGRID_TOPLEFT,  # Text position
            self.viewport,  # Viewport
            "IK Solver",  # Primary text
            "3D 6DoF Robot Arm",  # Secondary text
            self.context  # Context
        )

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def simulate(self):
        while not glfw.window_should_close(self.window):
            self.step() 

        glfw.terminate()
        
    def cursor_pos_callback(self, window, xpos, ypos):
        # Update the mouse position in window space
        self.mouse_x = xpos
        self.mouse_y = ypos
        print(f"Mouse position: {self.mouse_x}, {self.mouse_y}")

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_LEFT:
                self.robot.move_target(np.array([0, 0.01, 0]), "translate")  # Move target up
            elif key == glfw.KEY_RIGHT:
                self.robot.move_target(np.array([0, -0.01, 0]))
            elif key == glfw.KEY_DOWN:
                self.robot.move_target(np.array([-0.002, 0, 0]))
            elif key == glfw.KEY_UP:
                self.robot.move_target(np.array([0.002, 0, 0]))
            elif key == glfw.KEY_O:
                self.robot.move_target(np.array([0, 0, 0.01]))
            elif key == glfw.KEY_L:
                self.robot.move_target(np.array([0, 0, -0.01]))
            elif key == glfw.KEY_Q: 
                self.robot.move_target(np.array([-0.1, 0, 0]), "rotate")
            elif key == glfw.KEY_W:
                self.robot.move_target(np.array([0.1, 0, 0]), "rotate")
            elif key == glfw.KEY_A:
                self.robot.move_target(np.array([0, -0.1, 0]), "rotate")
            elif key == glfw.KEY_S:
                self.robot.move_target(np.array([0, 0.1, 0]), "rotate")
            elif key == glfw.KEY_Z:
                self.robot.move_target(np.array([0, 0, -0.1]), "rotate")
            elif key == glfw.KEY_X:
                self.robot.move_target(np.array([0, 0, 0.1]), "rotate")

        # print(self.robot.data.xpos[self.robot.target_id])
                


if __name__ == "__main__":
    robot_interface = Interface("mjcf/3d_6dof_robot.xml")
    robot_interface.robot.chain.add_eef_offset(np.array([0.051, 0, 0]))
    robot_interface.simulate()

    # Create a kinematic chain 

    # Read x, y of the target position (interface)

    # Compute IK: x, y -> theta1, theta2 (kinematics)

    # Move the robot to the target position (interface)

    # Visualize 

    # Print out thetas and error(np.linalg.norm(target_pos - current_pos))

    # Repeat 