import numpy as np 
from typing import List
from transform_utils import * 

class Frame: 
    def __init__(self, joint, rpy): 
        self.joint = joint 
        self.rpy = rpy 

    def get_transform(self, theta): 
        ''' 
        Assume all joints are revolute joints

        Args: 
            - theta: angle of rotation along the axis of rotation 

        Returns: 
            - T: 4x4 homogeneous transformation matrix [[R p][0 1]]
        '''
        T_local = np.eye(4)
        R = rot_from_axisangle(self.joint.axis, theta) 
        T_local[:3, :3] = R 
        T_local[:3, 3] = self.joint.offset

        return T_local 


class Joint: 
    def __init__(self, axis, offset): 
        self.axis = axis 
        self.offset = offset # xyz 


class KinematicChain: 
    def __init__(self, frames: List[Frame]): 
        self.transforms_world = [] 
        self.frames = frames

    @property 
    def transforms_local(self): 
        transforms = [] 
        for joint in self.joints: 
            transforms.append(joint.T_joint)
        return transforms 

    # @property 
    # def current_thetas(self): 
    #     return [joint.rpy for joint in self.joints]
    
    def set_joints(self, thetas): 
        for joint, new_rpy in zip(self.joints, thetas): 
            joint.rpy = new_rpy

    def compute_world_transforms(self):
        '''
        Update world-frame transformations
        '''
        self.transforms_world = [] 

        T_w = np.eye(4)
        for T_local in self.transforms_local:
            T_w = T_w @ T_local
            self.transforms_world.append(T_w)


    def forward_kinematics(self, frames: List[Frame], thetas: np.ndarray) -> np.ndarray:
        '''
        TODO: return the angle as well (ZYX and Axis-angle) 
        Args:
            frames: (list or Frame()) robot's frame for forward kinematics
            thetas: (sequence of float) input joint angles

        Returns:
            - xyz_eef: (3,) current xyz position of the end-effector 
        '''

        # Iterate over all frames 
        # Create new data structure fk that holds the Homo Mat w.r.t. the world for each frame 
        assert len(frames) == len(thetas), "shape mismatch"
        fk = []
        T_i_w = np.eye(4) # inital

        for frame, theta in zip(frames, thetas): 
            T_i_local = frame.get_transform(theta)
            T_i_w = T_i_w @ T_i_local
            fk.append(T_i_w) 
        
        return fk 
    
    def calc_jacobian(self, frames, cur_fk): 
        J = np.zeros((6, len(self.transforms_local))) # Shape: (6, n)
        p_eff = cur_fk[-1][:3, 3]

        for idx, frame, trans in enumerate(zip(frames, cur_fk)): 
            # Extract R_w_i and p_i from cur_fk 
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

        J_i = np.hstack([J_v, J_w])
        return J_i

    def compute_ik_newton_rapshon(self, current_thetas, target_pos, max_iter=100): 
        '''
        Args: 
            - current_thetas    : (n,) needed to compute the error (TODO: Positional error for now; later I will add rotational error as well)
            - target_pos        : (6,) target_xyz target_rpy 

        Returns: 
            - thetas            : target joint values 
        '''
        iterator = 1
        eps = float(1e-6)
        current_xyz = self.forward_kinematics()
        current_pos = np.concatenate([current_xyz, np.zeros(3)]) # (3,) -> (6,) just padding with zero to match 6D spatial coord

        # NOTE: positional error only for now 
        err_pos = np.array(target_pos - current_pos)
        err = np.linalg.norm(err_pos)

        while err > eps: 
            if iterator > max_iter: 
                break 
            
            # Jacobian 
            J = self.calc_jacobian()

            dtheta = np.linalg.pinv(J) @ err_pos 
            print(1, current_thetas)
            print(2, dtheta)
            current_thetas += dtheta 
            self.set_joints(current_thetas)

            current_xyz = self.forward_kinematics()
            current_pos = np.concatenate([current_xyz, np.zeros(3)]) # (3,) -> (6,) just padding with zero to match 6D spatial coord
            err_pos = np.array(target_pos - current_pos)

            err = np.linalg.norm(err_pos)
            iterator += 1 

        
        print(f"Total iteration: {iterator-1}")
        print(f"Error: {err}")

        return current_thetas  



def compute_homo_ts(rpy, xyz): 
    assert rpy.shape == (3,)
    assert xyz.shape == (3,) 

    T = np.eye(4) 
    T[:3, :3] = rot_from_rpy(rpy)
    T[:3, 3] = xyz
    return T 


if __name__ == "__main__": 
    ######## Unit Test ########
    joint1 = Joint(np.array([0, 0, 1]), np.array([3, 0, 0]), np.array([np.pi/6, np.pi/4, np.pi/3])) # Joint(axis, xyz, rpy)
    joint2 = Joint(np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 0])) 
    chain1 = KinematicChain([joint1, joint2])

    # Test 1 
    print("--- Test 1 ---")
    xyz_eef = chain1.forward_kinematics()
    print(chain1.forward_kinematics())
    print("Jacobian 6x2", chain1.calc_jacobian())
    assert np.all(np.isclose(xyz_eef, (3.354, 0.612, -0.707), rtol=0.001))

    # Test 2: change joint1's rpy
    print("\n--- Test 2 ---")
    chain1.joints[0].rpy = np.array([np.pi/6, 0, 0])

    print(chain1.forward_kinematics())
    print("Jacobian 6x2", chain1.calc_jacobian())
    
    ######## IK Test ########
    # print("2D Robot IK Test")
    # j0 = Joint(np.array([0, 0, 1]), np.array([0, 0, 0]), np.array([0, 0, 0])) # Joint(axis, xyz, rpy)
    # j1 = Joint(np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([0, 0, 0])) 
    # eef = Joint(np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([0, 0, 0])) # NOTE: EEF is represented with Joint class but it's actually just a stationary joint

    # robot2DoF = KinematicChain([j0, j1, eef])

    # # test 
    # current_xyz = robot2DoF.forward_kinematics() 
    # current_thetas = robot2DoF.current_thetas
    # print(f"Current xyz: {current_xyz}")
    # print(f"Current thetas: {current_thetas}")
    # assert np.all(current_xyz == (0, 2, 0)), f"FK is wrong, the current xyz is {current_xyz}"

    # target_pos = np.array([1, 1, 0, 0, 0, 0]) # ignore the rotation part 
    # new_thetas = robot2DoF.compute_ik_newton_rapshon(current_thetas, target_pos)
    # print(new_thetas)


    

