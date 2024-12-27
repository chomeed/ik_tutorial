'''
You may verify the code from: https://dugas.ch/transform_viewer/index.html
'''
import numpy as np  

def rpy_to_axisangle():
    pass 

def axisangle_to_rpy(): 
    pass 

def rot_from_rpy(rpy: np.ndarray) -> np.ndarray:
    '''
    Yaw-Pitch-Roll order 
    ZYX 
    R_z @ R_y @ R_x
    '''
    R = np.eye(3)

    for axis, angle in enumerate(rpy): 
        if np.isclose(angle, 0.0): 
            continue 

        c = np.cos(angle)
        s = np.sin(angle) 

        if axis == 0:       # rotate about x-axis 
            R_i = np.array([
                [1,  0,  0],
                [0,  c, -s],
                [0,  s,  c]
                ])
        elif axis == 1:     # rotate about y-axis
            R_i = np.array([
                [ c,  0,  s],
                [ 0,  1,  0],
                [-s,  0,  c]
                ])
        else:               # rotate about z-axis
            R_i = np.array([
                [c, -s,  0], 
                [s,  c,  0],
                [0,  0,  1]
                ])
        R = R_i @ R

    return R 

def rpy_from_rot(R: np.ndarray) -> np.ndarray:
    '''
    Yaw-Pitch-Roll order 
    ZYX 
    R_z @ R_y @ R_x
    '''
    rpy = np.zeros(3)
    rpy[0] = np.arctan2(R[1, 0], R[0, 0]) # yaw 
    rpy[1] = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)) # pitch 
    rpy[2] = np.arctan2(R[2, 1], R[2, 2]) # roll 

    return rpy

def axisangle_from_rot(R): 
    '''
    Reference: https://en.wikipedia.org/wiki/Axis–angle_representation
    '''
    trace_R = np.trace(R) 
    angle = np.arccos((trace_R - 1)/2)
    axis = (1 / (2*np.sin(angle))) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]).T

    return axis, angle 

def rot_from_axisangle(axis, angle): 
    '''
    Reference: https://gaussian37.github.io/vision-concept-axis_angle_rotation/
    '''
    n_x, n_y, n_z = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle) 
    t = 1 - c

    R = np.array([
        [t*n_x**2 + c, t*n_x*n_y - s*n_z, t*n_z*n_x + s*n_y],
        [t*n_x*n_y + s*n_z, t*n_y**2 +c, t*n_y*n_z - s*n_x],
        [t*n_x*n_z - s*n_y, t*n_y*n_z + s*n_x, t*n_z**2 + c]
    ])

    return R 


if __name__=="__main__": 
    axis = np.array([0.26726124, 0.53452248, 0.80178373])
    angle = np.pi/2 # 60 degrees 

    R = rot_from_axisangle(axis, angle) 
    print(R)

    axis, angle = axisangle_from_rot(R)
    print("axis", axis)
    print("angle", angle)