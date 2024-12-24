# TODO 1: Compute Jacobian (done)
# TODO 2: Solve 2D IK using Jacobian Transpose Method (done) 
# TODO 3: Compute Jacobian Pseudo-Inverse (done)
# TODO 4: Solve 2D IK using Jacobian Pseudo-Inverse Method (done) 

import numpy as np 
from math import sin, cos

def compute_jacobian(L1, L2, theta1, theta2):
    '''
    Arguments:
        - L1, L2: link lengths (unit of length)
        - theta1, theta2: joint angles (radians)
    
    Returns: 
        - J: 4x4 Jacobian matrix 
    '''
    J = np.zeros((2, 2))
    J[0, 0] = -L1*sin(theta1) - L2*sin(theta1+theta2)
    J[0, 1] = -L2*sin(theta1+theta2)
    J[1, 0] = L1*cos(theta1) + L2*cos(theta1+theta2)
    J[1, 1] = L2*cos(theta1+theta2)
    return J

def solve_2d_ik_jacobian(target_x, target_y, current_x, current_y, J, method='transpose'): 
    '''
    Arguments: 
        - target_x
        - target_y
        - current_x
        - current_y
        - J: 4x4 Jacobian matrix 
        
    Returns: 
        - delta_q 
    '''

    delta_pos = np.array([target_x - current_x, target_y - current_y])
    
    if method == "transpose": 
        J_transpose = J.T
        delta_pos *= 0.1 
    elif method == "pseudoinverse": 
        J_transpose = np.linalg.pinv(J)
    else: 
        raise ValueError("Method not set properly.")
    
    delta_q = J_transpose @ delta_pos 
    return delta_q 


if __name__=="__main__":
    from math import pi
    
    # initial setting 
    L1 = 3
    L2 = 2
    current_theta1 = pi/6
    current_theta2 = pi/4
    target_x = 4 
    target_y = 2 
    current_x = L1*cos(current_theta1) + L2*cos(current_theta1+current_theta2)
    current_y = L1*sin(current_theta1) + L2*sin(current_theta1+current_theta2)

    print(f"Current x: {current_x}")
    print(f"Current y: {current_y}")

    J = compute_jacobian(L1, L2, current_theta1, current_theta2)
    delta_q = solve_2d_ik_jacobian(target_x, target_y, current_x, current_y, J, method="pseudoinverse") # choose between 'tranpose' and 'pseudoinverse' 

    print(f"Delta q: {delta_q}")

    current_q = np.array([current_theta1, current_theta2])
    new_q = current_q + delta_q 

    print(f"New q: {new_q}")

    new_x = L1*cos(new_q[0]) + L2*cos(new_q[0]+new_q[1])
    new_y = L1*sin(new_q[0]) + L2*sin(new_q[0]+new_q[1])

    print(f"New pose: [{new_x}, {new_y}]")