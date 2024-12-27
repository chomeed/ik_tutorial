import numpy as np 
from typing import List
from transform_utils import * 


class Joint: 
    def __init__(self, axis, xyz, rpy): 
        self.axis = axis 
        
        R = rot_from_rpy(rpy) 
        T_offset = np.eye(4) 
        T_offset[:3, :3] = R
        T_offset[:3, 3] = xyz

        self.T_offset = T_offset 


class Frame: 
    def __init__(self, joint: Joint): 
        ''' 
        joint -> axis, T_offset, angle (input)
        
        '''
        self.joint = joint 

    def get_transform(self, theta): 
        ''' 
        Assume all joints are revolute joints

        Args: 
            - theta: angle of rotation along the axis of rotation 

        Returns: 
            - T: 4x4 homogeneous transformation matrix [[R p][0 1]]
        '''
        T_local = np.eye(4)
        R = rot_from_axisangle(self.joint.axis, theta) # rotation due to theta

        # where is offset? where does "rpy" play a role? 
        T_local[:3, :3] = R 

        return self.joint.T_offset @ T_local 
    
    def __str__(self): 
        return f"Joint: {self.joint.axis} {self.joint.T_offset}"


class KinematicChain: 
    def __init__(self, frames: List[Frame]): 
        self.transforms_world = [] # should be equal to fk 
        self.frames = frames

    @property 
    def transforms_local(self): 
        transforms = [] 
        for joint in self.joints: 
            transforms.append(joint.T_joint)
        return transforms 

    def compute_world_transforms(self):
        '''
        Update world-frame transformations
        '''
        self.transforms_world = [] 

        T_w = np.eye(4)
        for T_local in self.transforms_local:
            T_w = T_w @ T_local
            self.transforms_world.append(T_w)


    def forward_kinematics(self, thetas: np.ndarray) -> np.ndarray:
        '''
        TODO: return the angle as well (ZYX and Axis-angle) 
        Args:
            thetas: input joint angles

        Returns:
            - xyz_eef: (3,) current xyz position of the end-effector 
        '''
        # Iterate over all frames 
        # Create new data structure fk that holds the Homo Mat w.r.t. the world for each frame 
        frames = self.frames

        assert len(frames) == len(thetas), "shape mismatch"
        fk = []
        T_i_w = np.eye(4) # inital

        for frame, theta in zip(frames, thetas): 
            T_i_local = frame.get_transform(theta)
            T_i_w = T_i_w @ T_i_local
            fk.append(T_i_w) 
        
        return fk 
    
    def calc_jacobian(self, curr_fk): 
        frames = self.frames
        # curr_fk = self.forward_kinematics(frames, thetas)

        J = np.zeros((3, len(self.frames))) # Shape: (6, n) NOTE: self.frames -> dof 
        p_eff = curr_fk[-1][:3, 3]

        for idx, (frame, trans) in enumerate(zip(frames, curr_fk)): 
            # Extract R_w_i and p_i from curr_fk 
            # Extract joint axis from frame.joint 
            R_w_i = trans[:3, :3]
            p_i = trans[:3, 3]
            w_i = frame.joint.axis 
            J_i = self._compute_jacobian_column(w_i, R_w_i, p_eff, p_i)
            J[:, idx] = J_i 
        
        return J

    def _compute_jacobian_column(self, w_i, R_w_i, p_eff, p_i): 
        '''
        Assumptions: 
            - only one link 

        Args: 
            - w_i       : axis of rotation w.r.t the joint frame
            - R_w_i     : rotation matrix from joint frame to world frame 
            - p_eff     : end effector position in world coordinates 
            _ p_i       : joint position in world coordinates 

        Returns:   
            - J_i: Jacobian 6x1 column vector [J_w, J_v]
        '''
        J_w = R_w_i @ w_i 
        displacement = p_eff - p_i
        J_v = np.cross(J_w, displacement) 

        J_i = J_v   # only position in consideration 
        # J_i = np.hstack([J_v, J_w]) # original 
        return J_i

    def compute_ik_newton_rapshon(self, current_thetas: np.ndarray, target_pos, max_iter=1000): 
        '''
        Args: 
            - current_thetas    : (n,) needed to compute the error (TODO: Positional error for now; later I will add rotational error as well)
            - target_pos        : (6,) target_xyz target_rpy 

        Returns: 
            - thetas            : target joint values 
        '''
        iterator = 1
        eps = float(1e-6)
        curr_fk = self.forward_kinematics(current_thetas)
        current_xyz = curr_fk[-1][:3, 3]
        current_pos = np.concatenate([current_xyz]) # (3,) -> (6,) just padding with zero to match 6D spatial coord

        # NOTE: positional error only for now 
        err_pos = np.array(target_pos - current_pos)
        err = np.linalg.norm(err_pos)
        print("ERROR pos", err_pos)

        while err > eps: 
            if iterator > max_iter: 
                break 
            
            # Jacobian 
            J = self.calc_jacobian(curr_fk)

            dtheta = 0.5 * np.linalg.pinv(J) @ err_pos 
            # print("current thetas", current_thetas)
            # print("dtheta", dtheta)
            current_thetas += dtheta 

            curr_fk = self.forward_kinematics(current_thetas)
            current_xyz = curr_fk[-1][:3, 3]
            current_pos = np.concatenate([current_xyz]) # (3,) -> (6,) just padding with zero to match 6D spatial coord
            err_pos = np.array(target_pos - current_pos)

            err = np.linalg.norm(err_pos)
            iterator += 1 

        
        print(f"Total iteration: {iterator-1}")
        print(f"Error: {err}")

        info = {
            "error": err,
            "iterations": iterator-1,
            "target_pos": target_pos,
            "current_pos": current_pos
            }

        return current_thetas, info 


if __name__ == "__main__": 
    ######## Unit Test ########
    # joint1 = Joint(np.array([0, 0, 1]), np.array([3, 0, 0]), np.array([np.pi/6, np.pi/4, np.pi/3])) # Joint(axis, xyz, rpy)
    # joint2 = Joint(np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 0])) 

    # frame1 = Frame(joint1)
    # frame2 = Frame(joint2) 

    # chain1 = KinematicChain([frame1, frame2]) 

    # # Test 1 
    # print("--- Test 1 ---")
    # curr_fk = chain1.forward_kinematics(np.zeros(2))
    # print("XYZ EEF", curr_fk[-1][:3, 3])
    # print("Jacobian 6x2", chain1.calc_jacobian(curr_fk))
    # assert np.all(np.isclose(curr_fk[-1][:3, 3], (3.354, 0.612, -0.707), rtol=0.001))

    # # # Test 2: change joint1's rpy
    # print("\n--- Test 2 ---")
    # print("rotate first joint by +pi/6")
    # thetas = np.array([np.pi/6, 0])
    # curr_fk = chain1.forward_kinematics(thetas)
    # print("XYZ EEF", curr_fk[-1][:3, 3])
    # print("Jacobian 6x2", chain1.calc_jacobian(curr_fk))
    
    ######## IK Test ########
    print("--- 2D Robot IK Test ---")
    j0 = Joint(np.array([0, 0, 1]), np.array([0, 0, 0]), np.array([0, 0, 0])) # Joint(axis, xyz, rpy)
    j1 = Joint(np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([0, 0, 0])) 
    eef = Joint(np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([0, 0, 0]))

    frame1 = Frame(j0)
    frame2 = Frame(j1) 
    frame3 = Frame(eef)

    robot2DoF = KinematicChain([frame1, frame2, frame3]) 


    # test 
    current_thetas = np.zeros(3)
    curr_fk = robot2DoF.forward_kinematics(current_thetas)
    current_xyz = curr_fk[-1][:3, 3]    

    print(f"Current xyz: {current_xyz}")
    print(f"Current thetas: {current_thetas}")
    assert np.all(current_xyz == (0, 2, 0)), f"FK is wrong, the current xyz is {current_xyz}"

    target_pos = np.array([1, 1, 0]) # ignore the rotation part 
    new_thetas, _ = robot2DoF.compute_ik_newton_rapshon(current_thetas, target_pos)
    print("New thetas", new_thetas)

    new_fk = robot2DoF.forward_kinematics(new_thetas)
    new_xyz = new_fk[-1][:3, 3]  
    print("New xyz", new_xyz)


    

