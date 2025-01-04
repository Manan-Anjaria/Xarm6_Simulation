import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pybullet as p
import pybullet_data
import time
import random

def transpose(matrix):
    """
    Transpose of a matrix.
    """
    transposed = []
    for i in range(len(matrix[0])):
        transposed_row = []
        for row in matrix:
            transposed_row.append(row[i])
        transposed.append(transposed_row)
    return transposed

def matrix_multiply(a, b):
    """
    Multiply two matrices a and b.
    """
    result = []
    for i in range(len(a)):
        result_row = []
        for j in range(len(b[0])):
            sum_elements = 0
            for k in range(len(b)):
                sum_elements += a[i][k] * b[k][j]
            result_row.append(sum_elements)
        result.append(result_row)
    return result

def cross_product(a, b):
    """
    Compute the cross product of two 3D vectors a and b.
    """
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]

def determinant(matrix):
    """
    Compute the determinant of a square matrix using recursion.
    """
    if len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be square to compute determinant.")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        # Base case for 2x2 matrix
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    
    det = 0
    for c in range(len(matrix)):
        minor = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1)**c) * matrix[0][c] * determinant(minor)
    return det

def matrix_inverse(matrix):
    """
    Compute the inverse of a square matrix using Gaussian elimination.
    Assumes matrix is square and invertible.
    """
    n = len(matrix)
    # Create an augmented matrix with the identity matrix
    augmented = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(matrix)]
    
    # Perform Gaussian elimination
    for i in range(n):
        # Find the pivot
        max_row = i
        max_val = abs(augmented[i][i])
        for k in range(i+1, n):
            if abs(augmented[k][i]) > max_val:
                max_val = abs(augmented[k][i])
                max_row = k
        if max_val == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
        # Swap rows if needed
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        # Normalize the pivot row
        pivot = augmented[i][i]
        augmented[i] = [element / pivot for element in augmented[i]]
        # Eliminate the other rows
        for j in range(n):
            if j != i:
                factor = augmented[j][i]
                augmented[j] = [augmented[j][k] - factor * augmented[i][k] for k in range(2*n)]
    
    # Extract the inverse matrix
    inverse = [row[n:] for row in augmented]
    return inverse

def pseudo_inverse(matrix):
    """
    Compute the pseudo-inverse of a matrix using the formula A^+ = (A^T A)^-1 A^T
    """
    A_T = transpose(matrix)
    A_TA = matrix_multiply(A_T, matrix)
    try:
        A_TA_inv = matrix_inverse(A_TA)
    except ValueError:
        raise ValueError("Matrix A^T A is singular, cannot compute pseudo-inverse.")
    A_pseudo_inv = matrix_multiply(A_TA_inv, A_T)
    return A_pseudo_inv

class XArm6Robot:
    # Define offsets for joints 2 and 3 in radians
    T2_offset = -math.atan(284.5 / 53.5)  # ≈ -1.3849179 radians
    T3_offset = -T2_offset                # ≈ 1.3849179 radians

    def __init__(self, joint_angles_deg, joint_limits_deg):
        """
        Initialize the XArm6 Robot with joint angles and joint limits.

        Parameters:
        - joint_angles_deg (list): Initial joint angles in degrees.
        - joint_limits_deg (list of tuples): Joint limits as (min_deg, max_deg) for each joint.
        """
        self.joint_angles_deg = [float(angle) for angle in joint_angles_deg]  # Store joint angles in degrees
        self.joint_angles_rad = [math.radians(angle) for angle in self.joint_angles_deg]  # Convert to radians for computation
        self.joint_limits_deg = joint_limits_deg                          # Store joint limits
        self.transformation_matrices = []                                # To store transformation matrices
        self.joint_positions = []                                        # To store positions of joints
        self.joint_axes = []                                             # To store joint axes
        self.end_effector_matrix = None                                  # Transformation matrix of end effector
        self.jacobian = None                                             # Jacobian matrix
        self.compute_forward_kinematics()                                # Compute FK on initialization

    def dh_transformation(self, theta, a, d, alpha):
        """
        Compute the Denavit-Hartenberg transformation matrix.

        Parameters:
        - theta (float): Joint angle in radians.
        - a (float): Link length.
        - d (float): Link offset.
        - alpha (float): Link twist.

        Returns:
        - list of lists: 4x4 transformation matrix.
        """
        return [
            [math.cos(theta), -math.sin(theta)*math.cos(alpha),  math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
            [math.sin(theta),  math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
            [0,               math.sin(alpha),                 math.cos(alpha),                d],
            [0,               0,                               0,                              1]
        ]  # Create the DH matrix based on the parameters

    def compute_forward_kinematics(self):
        """
        Compute the forward kinematics for the robot, considering offsets for joints 2 and 3.
        """
        # Define DH parameters: [theta, a, d, alpha] for each joint
        # Apply offsets to joints 2 and 3
        dh_parameters = [
            [self.joint_angles_rad[0], 0,    267,  -math.pi/2],                             # Joint 1
            [self.joint_angles_rad[1] + self.T2_offset, 290,    0,   0],                  # Joint 2 with offset
            [self.joint_angles_rad[2] + self.T3_offset, 77.5,    0,  -math.pi/2],          # Joint 3 with offset
            [self.joint_angles_rad[3], 0,  343,     math.pi/2],                            # Joint 4
            [self.joint_angles_rad[4], 76,  0,  -math.pi/2],                              # Joint 5
            [self.joint_angles_rad[5],0,    97,            0]                           # Joint 6
        ]

        current_transform = [
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ]  # Start with identity matrix
        self.transformation_matrices = [current_transform.copy()]  # Reset transformation matrices
        # Base position
        self.joint_positions = [
            [current_transform[0][3], current_transform[1][3], current_transform[2][3]]
        ]  
        # Base axis (z-axis)
        self.joint_axes = [
            [current_transform[0][2], current_transform[1][2], current_transform[2][2]]
        ]

        # Iterate through each joint to compute the transformation matrices
        for idx, params in enumerate(dh_parameters):
            theta, a, d, alpha = params
            transform = self.dh_transformation(theta, a, d, alpha)  # Get the DH matrix
            current_transform = matrix_multiply(current_transform, transform)  # Multiply to get current transform
            self.transformation_matrices.append([row[:] for row in current_transform])  # Store it
            # Extract joint position
            joint_pos = [current_transform[0][3], current_transform[1][3], current_transform[2][3]]
            self.joint_positions.append(joint_pos)
            # Extract joint axis (z-axis of the current frame)
            joint_axis = [current_transform[0][2], current_transform[1][2], current_transform[2][2]]
            self.joint_axes.append(joint_axis)

        self.end_effector_matrix = [row[:] for row in current_transform]  # Final transform is end effector
        self.compute_jacobian()  # Compute Jacobian after FK

    def compute_jacobian(self):
        """
        Compute the Jacobian matrix for the robot.
        """
        # Initialize a 6x6 Jacobian matrix filled with zeros
        J = [[0 for _ in range(6)] for _ in range(6)]
        end_pos = self.get_end_effector_position()  # Position of end effector

        for i in range(6):
            zi = self.joint_axes[i]  # Axis of the i-th joint
            pi = self.joint_positions[i]  # Position of the i-th joint
            # Compute the linear velocity part (cross product of zi and (end_pos - pi))
            end_minus_pi = [end_pos[0] - pi[0], end_pos[1] - pi[1], end_pos[2] - pi[2]]
            Jp = cross_product(zi, end_minus_pi)
            Jo = zi  # Angular velocity part is the joint axis itself
            # Assign to Jacobian matrix
            J[0][i] = Jp[0]
            J[1][i] = Jp[1]
            J[2][i] = Jp[2]
            J[3][i] = Jo[0]
            J[4][i] = Jo[1]
            J[5][i] = Jo[2]

        self.jacobian = J  # Store the Jacobian

    def get_end_effector_position(self):
        """
        Get the end effector position.

        Returns:
        - list: 3-element position vector.
        """
        return [self.end_effector_matrix[0][3], self.end_effector_matrix[1][3], self.end_effector_matrix[2][3]]  # Extract position from transformation matrix

    def get_joint_positions(self):
        """
        Get the positions of all joints.

        Returns:
        - list of lists: List of joint position vectors.
        """
        return self.joint_positions  # Return list of joint positions

    def get_jacobian(self):
        """
        Get the Jacobian matrix.

        Returns:
        - list of lists: 6x6 Jacobian matrix.
        """
        return self.jacobian  # Return the computed Jacobian

    @staticmethod
    def adjoint(transformation_matrix):
        """
        Compute the adjoint of a transformation matrix.

        Parameters:
        - transformation_matrix (list of lists): 4x4 transformation matrix.

        Returns:
        - list of lists: 6x6 adjoint matrix.
        """
        R = [row[:3] for row in transformation_matrix[:3]]  # Rotation part
        p = [transformation_matrix[0][3], transformation_matrix[1][3], transformation_matrix[2][3]]    # Position part
        # Create skew-symmetric matrix for position vector
        p_skew = [
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]
        ]
        # Block matrix for adjoint
        adj = [
            [R[0][0], R[0][1], R[0][2], 0,      0,      0],
            [R[1][0], R[1][1], R[1][2], 0,      0,      0],
            [R[2][0], R[2][1], R[2][2], 0,      0,      0],
            [p_skew[0][0], p_skew[0][1], p_skew[0][2], R[0][0], R[0][1], R[0][2]],
            [p_skew[1][0], p_skew[1][1], p_skew[1][2], R[1][0], R[1][1], R[1][2]],
            [p_skew[2][0], p_skew[2][1], p_skew[2][2], R[2][0], R[2][1], R[2][2]]
        ]
        return adj  # Return adjoint matrix

    def inverse_dynamics(self, position, force):
        """
        Compute joint torques (inverse dynamics) based on position and force vectors.

        Parameters:
        - position (list): Position vector [px, py, pz].
        - force (list): Force vector [Fx, Fy, Fz].

        Returns:
        - list: Joint torques [tau_x, tau_y, tau_z].
        """
        torque = cross_product(position, force)  # Calculate torque as cross product
        return torque  # Return the torque

    def enforce_joint_limits(self):
        """
        Enforce joint limits on the current joint angles.
        """
        for i in range(len(self.joint_angles_deg)):
            min_angle, max_angle = self.joint_limits_deg[i]  # Get limits for joint i
            if self.joint_angles_deg[i] < min_angle:
                print(f"Joint {i+1} angle {self.joint_angles_deg[i]:.2f}° below limit. Clamping to {min_angle}°.")
                self.joint_angles_deg[i] = min_angle  # Clamp to min
            elif self.joint_angles_deg[i] > max_angle:
                print(f"Joint {i+1} angle {self.joint_angles_deg[i]:.2f}° above limit. Clamping to {max_angle}°.")
                self.joint_angles_deg[i] = max_angle  # Clamp to max
        self.joint_angles_rad = [math.radians(angle) for angle in self.joint_angles_deg]  # Update radians after clamping

    def inverse_kinematics(self, target_pos, max_iterations=1000, threshold=0.01, damping=0.1):
        """
        Perform inverse kinematics using the Newton-Raphson method with damping and joint limits.

        Parameters:
        - target_pos (list): Target end effector position (3,).
        - max_iterations (int): Maximum number of iterations.
        - threshold (float): Convergence threshold for the error norm.
        - damping (float): Damping factor for the pseudo-inverse.

        Returns:
        - list: Computed joint angles in degrees.
        - list: History of error norms.
        """
        error_history = []  # To store error norms
        for iteration in range(max_iterations):
            self.compute_forward_kinematics()  # Update FK
            current_pos = self.get_end_effector_position()  # Current end effector pos
            position_error = [target_pos[i] - current_pos[i] for i in range(3)]  # Error vector
            error_norm = math.sqrt(sum([error**2 for error in position_error]))  # Error magnitude
            error_history.append(error_norm)  # Record error

            if error_norm < threshold:
                print(f"\nInverse Kinematics Converged in {iteration} iterations.")
                return self.joint_angles_deg, error_history  # Success

            J = self.get_jacobian()  # Get Jacobian
            # Extract the linear velocity part of the Jacobian (first 3 rows)
            J_velocity = [row[:6] for row in J[:3]]  # 3x6 matrix

            # Compute J_velocity^T * J_velocity
            J_T = transpose(J_velocity)  # 6x3
            J_TJ = matrix_multiply(J_T, J_velocity)  # 6x6 matrix

            # Add damping to the diagonal
            for i in range(len(J_TJ)):
                J_TJ[i][i] += damping ** 2

            # Compute pseudo-inverse: (J^T J)^-1 J^T
            try:
                J_TJ_inv = matrix_inverse(J_TJ)  # Invert J_TJ
            except ValueError:
                print("J_TJ matrix is singular, cannot compute inverse.")
                break
            J_pseudo_inverse = matrix_multiply(J_TJ_inv, J_T)  # 6x3 matrix

            # Compute delta_theta = J_pseudo_inverse * position_error
            delta_theta_rad = [0 for _ in range(6)]
            for i in range(6):
                for j in range(3):
                    delta_theta_rad[i] += J_pseudo_inverse[i][j] * position_error[j]

            # Convert change in angles to degrees
            delta_theta_deg = [math.degrees(delta_theta_rad[i]) for i in range(6)]

            # Update joint angles
            self.joint_angles_deg = [self.joint_angles_deg[i] + delta_theta_deg[i] for i in range(6)]
            self.enforce_joint_limits()  # Ensure joint limits

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Error Norm = {error_norm:.4f}")  # Progress update

        print("\nInverse Kinematics did not converge within the maximum iterations.")
        return self.joint_angles_deg, error_history  # Return even if not converged

def plot_workspace(end_effector_positions):
    """
    Plot the workspace of the robot.

    Parameters:
    - end_effector_positions (list of lists): Array of end effector positions.
    """
    # Convert list of lists to separate lists for plotting
    x = [pos[0] for pos in end_effector_positions]
    y = [pos[1] for pos in end_effector_positions]
    z = [pos[2] for pos in end_effector_positions]

    fig = plt.figure(figsize=(12, 10))  # Bigger figure for clarity
    ax = fig.add_subplot(111, projection='3d')  # 3D plot
    ax.scatter(x, y, z, c='blue', marker='o', s=1, alpha=0.3)  # Plot points
    ax.set_xlabel('X-axis (mm)')
    ax.set_ylabel('Y-axis (mm)')
    ax.set_zlabel('Z-axis (mm)')
    ax.set_title('Full Workspace of XArm6 Robot')
    plt.show()  # Show the plot

def plot_joint_positions(joint_positions):
    """
    Plot the joint positions of the robot in 3D space.

    Parameters:
    - joint_positions (list of lists): List of joint position vectors.
    """
    # Convert list of lists to separate lists for plotting
    joint_positions = [[pos[0], pos[1], pos[2]] for pos in joint_positions]
    x = [pos[0] for pos in joint_positions]
    y = [pos[1] for pos in joint_positions]
    z = [pos[2] for pos in joint_positions]

    fig = plt.figure(figsize=(10, 8))  # Set figure size
    ax = fig.add_subplot(111, projection='3d')  # 3D plot

    # Plot joints as red dots
    ax.scatter(x, y, z, c='red', marker='o', s=50, label='Joints')

    # Connect joints with black lines to show structure
    ax.plot(x, y, z, c='black', linewidth=2, label='Robot Structure')

    ax.set_xlabel('X-axis (mm)')
    ax.set_ylabel('Y-axis (mm)')
    ax.set_zlabel('Z-axis (mm)')
    ax.set_title('Joint Positions of XArm6 Robot')
    ax.legend()  # Show legend
    plt.show()  # Display the plot

def plot_jacobian(jacobian_matrix):
    """
    Plot the Jacobian matrix as a heatmap.

    Parameters:
    - jacobian_matrix (list of lists): 6x6 Jacobian matrix.
    """
    # Convert list of lists to separate lists for plotting
    jacobian_matrix = [[float(cell) for cell in row] for row in jacobian_matrix]
    fig, ax = plt.subplots(figsize=(8, 6))  # Set figure size
    cax = ax.imshow(jacobian_matrix, cmap='viridis')  # Create heatmap
    fig.colorbar(cax)  # Add color bar

    # Annotate each cell with its value using pure Python loops
    for i in range(len(jacobian_matrix)):
        for j in range(len(jacobian_matrix[0])):
            val = jacobian_matrix[i][j]
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    color='white' if abs(val) > 0.5 else 'black')

    ax.set_xticks(range(6))  # Set x ticks
    ax.set_yticks(range(6))  # Set y ticks
    ax.set_xticklabels([f'J{i+1}' for i in range(6)])  # Label joints
    ax.set_yticklabels([f'J{i+1}' for i in range(6)])  # Label joints
    ax.set_xlabel('Joint Index')
    ax.set_ylabel('Joint Index')
    ax.set_title('Jacobian Matrix Heatmap')
    plt.show()  # Show the plot

def plot_path(path_positions):
    """
    Plot the path followed by the end effector as a colored line.

    Parameters:
    - path_positions (list of lists): List of end effector positions.
    """
    # Convert list of lists to separate lists for plotting
    path_positions = [[pos[0], pos[1], pos[2]] for pos in path_positions]
    x = [pos[0] for pos in path_positions]
    y = [pos[1] for pos in path_positions]
    z = [pos[2] for pos in path_positions]

    fig = plt.figure(figsize=(10, 8))  # Set figure size
    ax = fig.add_subplot(111, projection='3d')  # 3D plot

    num_points = len(path_positions)
    # Create a color gradient based on the order of points
    colors = [plt.cm.jet(i/num_points) for i in range(num_points)]

    # Plot the path as a line with color gradient using pure Python loops
    for i in range(num_points -1):
        xs = [x[i], x[i+1]]
        ys = [y[i], y[i+1]]
        zs = [z[i], z[i+1]]
        ax.plot(xs, ys, zs, color=colors[i])

    ax.set_xlabel('X-axis (mm)')
    ax.set_ylabel('Y-axis (mm)')
    ax.set_zlabel('Z-axis (mm)')
    ax.set_title('Path Followed by End Effector')
    plt.show()  # Show the plot

def plot_forward_kinematics(robot_positions, target_position):
    """
    Plot the forward kinematics showing robot configuration and target position.
    
    Parameters:
    - robot_positions (list): List of joint positions including end effector
    - target_position (list): Target position for the end effector
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert positions to separate coordinates for plotting
    x_coords = [pos[0] for pos in robot_positions]
    y_coords = [pos[1] for pos in robot_positions]
    z_coords = [pos[2] for pos in robot_positions]
    
    # Plot robot links
    ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=2, label='Robot Links')
    
    # Plot joints
    ax.scatter(x_coords[:-1], y_coords[:-1], z_coords[:-1], 
              color='red', s=100, label='Joints')
    
    # Plot end effector
    ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
              color='green', s=100, label='End Effector')
    
    # Plot target position
    ax.scatter(target_position[0], target_position[1], target_position[2],
              color='purple', s=100, marker='*', label='Target')
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Forward Kinematics Configuration')
    ax.legend()
    
    plt.show()

def plot_error_norm_history(error_history):
    """
    Plot the history of error norms during inverse kinematics convergence.
    
    Parameters:
    - error_history (list): List of error norms from IK iterations
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(error_history)), error_history, 'b-', linewidth=2)
    plt.yscale('log')  # Use log scale for better visualization
    plt.xlabel('Iteration')
    plt.ylabel('Error Norm (mm)')
    plt.title('Inverse Kinematics Convergence')
    plt.grid(True)
    plt.show()

def visualize_simulation(initial_joint_angles_deg, final_joint_angles_deg, joint_limits_deg, urdf_path=r"C:\Users\anjar\OneDrive\Desktop\FOR\bullet3-master\examples\pybullet\gym\pybullet_data\xarm\xarm6_robot.urdf"):
   
    # Initialize PyBullet
    physics_client = p.connect(p.GUI)  # Start GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # To load plane and other objects
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])  # Set camera

    # Load the plane and robot
    plane_id = p.loadURDF("plane.urdf")  # Load a plane as the ground

    # Define the URDF path
    robot_urdf_path = urdf_path  # Use provided URDF path

    try:
        robot_id = p.loadURDF(robot_urdf_path, useFixedBase=True)  # Load robot URDF
    except Exception as e:
        print(f"Error: Unable to load URDF from path '{robot_urdf_path}'. Exception: {e}")
        p.disconnect()  # Disconnect if failed
        return

    # Set initial joint angles
    num_joints = p.getNumJoints(robot_id)  # Get number of joints
    for i in range(min(6, num_joints)):  # Assuming first 6 joints
        p.resetJointState(robot_id, i, math.radians(initial_joint_angles_deg[i]))  # Set joint angles

    # Function to interpolate between initial and target joint angles
    def interpolate_angles(initial, target, steps):
        """
        Linearly interpolate between initial and target joint angles.

        Parameters:
        - initial (list): Initial joint angles in radians.
        - target (list): Target joint angles in radians.
        - steps (int): Number of interpolation steps.

        Returns:
        - list of lists: Interpolated joint angles.
        """
        interpolated = []
        for step in range(steps):
            ratio = step / steps
            interpolated_angles = [initial[j] + (target[j] - initial[j]) * ratio for j in range(6)]
            interpolated.append(interpolated_angles)
        return interpolated

    # Number of simulation steps
    simulation_duration = 2  # seconds
    simulation_steps = 240 * simulation_duration  # Assuming 240 Hz

    # Interpolate joint angles from initial to final
    initial_angles_rad = [math.radians(angle) for angle in initial_joint_angles_deg]  # Convert to radians
    target_angles_rad = [math.radians(angle) for angle in final_joint_angles_deg]    # Convert to radians
    interpolated_angles = interpolate_angles(initial_angles_rad, target_angles_rad, simulation_steps)  # Get interpolated angles

    # Run the simulation
    for angles in interpolated_angles:
        for i in range(min(6, num_joints)):
            p.setJointMotorControl2(bodyIndex=robot_id,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=angles[i],
                                    force=500)  # Set joint positions
        p.stepSimulation()  # Step the simulation
        time.sleep(1/240)  # Sleep to match real-time

    # Keep the simulation window open for a while
    time.sleep(2)
    p.disconnect()  # Disconnect PyBullet

def main():
    # **Define joint limits in degrees as (min, max) for each joint**
    joint_limits_deg = [
        (-170, 170),  # Joint 1
        (-120, 120),  # Joint 2
        (-170, 170),  # Joint 3
        (-120, 120),  # Joint 4
        (-170, 170),  # Joint 5
        (-120, 120)   # Joint 6
    ]

    # **Define initial and final joint angles in degrees**
    initial_joint_angles_deg = [100, -10, -110, -50, -50, 50]  # Starting angles
    final_joint_angles_deg = [-70, -10, -10, -50, -50, 50]    # Ending angles

    # **Instantiate the robot with initial joint angles and joint limits**
    robot = XArm6Robot(initial_joint_angles_deg, joint_limits_deg)  # Create robot instance

    # **Display forward kinematics results**
    print("Joint Positions:")
    for idx, pos in enumerate(robot.get_joint_positions(), 1):
        print(f"Joint {idx}: {pos}")  # Print each joint position

    print("\nEnd Effector Position:", robot.get_end_effector_position())  # Print end effector pos
    print("\nJacobian Matrix:")
    for row in robot.get_jacobian():
        print(row)  # Print each row of the Jacobian

    # **Plot joint positions**
    plot_joint_positions(robot.get_joint_positions())  # Visualize joint positions

    # **Plot Jacobian matrix**
    plot_jacobian(robot.get_jacobian())  # Visualize Jacobian heatmap

    # **Generate and plot full workspace by random sampling**
    end_effector_positions = []
    num_samples = 10000  # Reduced number of random samples for computational efficiency

    print("\nGenerating full workspace with random sampling...")
    for _ in range(num_samples):
        random_angles_deg = [
            random.uniform(joint_limits_deg[i][0], joint_limits_deg[i][1]) for i in range(6)
        ]  # Random angles within limits in degrees
        temp_robot = XArm6Robot(random_angles_deg, joint_limits_deg)  # Temporary robot with random angles
        end_pos = temp_robot.get_end_effector_position()  # Get end effector position
        end_effector_positions.append(end_pos)  # Append to list

    plot_workspace(end_effector_positions)  # Plot the full workspace

    # **Define 7 waypoints for the end effector to reach**
    current_end_effector_pos = robot.get_end_effector_position()
    waypoints = [
        [current_end_effector_pos[0] + 260, current_end_effector_pos[1], current_end_effector_pos[2]],    # Move +X
        [current_end_effector_pos[0], current_end_effector_pos[1] + 230, current_end_effector_pos[2]],    # Move +Y
        [current_end_effector_pos[0], current_end_effector_pos[1], current_end_effector_pos[2] + 60],    # Move +Z
        [current_end_effector_pos[0] - 190, current_end_effector_pos[1], current_end_effector_pos[2]],   # Move -X
        [current_end_effector_pos[0], current_end_effector_pos[1] - 270, current_end_effector_pos[2]],   # Move -Y
        [current_end_effector_pos[0], current_end_effector_pos[1], current_end_effector_pos[2] - 60],   # Move -Z
        [current_end_effector_pos[0] + 140, current_end_effector_pos[1] + 60, current_end_effector_pos[2] + 80]   # Move diagonally
    ]

    print("\n7 Waypoints for the End Effector:")
    for idx, wp in enumerate(waypoints, 1):
        print(f"Waypoint {idx}: {wp}")  # Print each waypoint

    # **Initialize path_positions to store the path of the end effector**
    path_positions = [robot.get_end_effector_position()]  # Start with the initial position

    # **Perform Inverse Kinematics for each waypoint and simulate movement**
    for idx, target_position in enumerate(waypoints, 1):
        print(f"\nMoving to Waypoint {idx}: {target_position}")
        computed_joint_angles_deg, error_history = robot.inverse_kinematics(
            target_position,
            max_iterations=1000,
            threshold=0.01,
            damping=0.1
        )  # Compute IK for the waypoint
        print("Final Joint Angles after IK:", computed_joint_angles_deg)  # Print resulting angles
        #visualize_simulation(initial_joint_angles_deg, computed_joint_angles_deg, joint_limits_deg)  # Simulate movement
        initial_joint_angles_deg = computed_joint_angles_deg.copy()  # Update initial angles for next move
        path_positions.append(robot.get_end_effector_position())  # Append new position to path
    # After defining waypoints, add:
    print("\nComputing Forward and Inverse Kinematics for Waypoints...")
    
    all_error_histories = []  # Store error histories for all waypoints
    
    for idx, target_position in enumerate(waypoints, 1):
        print(f"\nMoving to Waypoint {idx}: {target_position}")
        
        # Get current robot configuration for FK plotting
        current_positions = robot.get_joint_positions()
        
        # Plot forward kinematics for current configuration and target
        plot_forward_kinematics(current_positions, target_position)
        
        # Compute IK
        computed_joint_angles_deg, error_history = robot.inverse_kinematics(
            target_position,
            max_iterations=1000,
            threshold=0.01,
            damping=0.1
        )
        
        # Plot error norm history
        print(f"Plotting error convergence for Waypoint {idx}")
        plot_error_norm_history(error_history)
        all_error_histories.append(error_history)
        
        print("Final Joint Angles after IK:", computed_joint_angles_deg)
        visualize_simulation(initial_joint_angles_deg, computed_joint_angles_deg, joint_limits_deg)
        initial_joint_angles_deg = computed_joint_angles_deg.copy()
        path_positions.append(robot.get_end_effector_position())
    # **Plot the path followed by the end effector**
    plot_path(path_positions)  # Visualize the path as a colored line

    # **Extra Credit Parts**
    print("\n***Extra Credit Parts***")

    # **Adjoint Matrix Example**
    # Desired end-effector transformation matrix (for demonstration)
    desired_T = [
        [43, 0, 0, 0],
        [0, 124, 0, 0],
        [0, 0, 43, 0],
        [0, 0, 0, 64]
    ]  # Identity matrix as example
    adj_T = XArm6Robot.adjoint(desired_T)  # Compute adjoint
    print("\nAdjoint Matrix of Desired Transformation:")
    for row in adj_T:
        print(row)  # Print adjoint matrix

    # **Inverse Dynamics Example**
    position = [1, 1, 1]  
    force = [5, 10, 0]    
    joint_torques = robot.inverse_dynamics(position, force)  # Compute torques
    print("\nComputed Joint Torques:", joint_torques)  # Print torques

    # **Compute Joint Torques using Jacobian**
    J_toolw = [row[:6] for row in robot.get_jacobian()[3:6]]  # Angular part of the Jacobian (3x6)
    # Transpose J_toolw to get 6x3
    J_toolw_T = transpose(J_toolw)  # 6x3
    # Convert joint_torques to a column vector
    force_cartesian = [[-joint_torques[i]] for i in range(3)]  # 3x1 matrix
    # Multiply J_toolw_T (6x3) with force_cartesian (3x1) to get (6x1)
    tau_joints_matrix = matrix_multiply(J_toolw_T, force_cartesian)  # 6x1
    # Flatten the result to a list
    tau_joints = [row[0] for row in tau_joints_matrix]
    print("\nJoint Torques from Jacobian:", tau_joints)  # Print joint torques

    # **Kineostatic Duality Implementation (Extra Credit)**
    print("\n***Kineostatic Duality (Extra Credit)***")
    # Forward Kinematics Duality: Mapping forces to joint torques
    # Assuming a force applied at the end effector
    applied_force = [7, 33, 5] 
    # Create a 6-element wrench (force and zero torque)
    wrench = applied_force + [0, 0, 0]  # Combine force and torque
    J = robot.get_jacobian()  # Get Jacobian
    # Transpose J to get 6x6
    J_T = transpose(J)  # 6x6
    # Multiply J_T (6x6) with wrench (6x1) to get joint torques (6x1)
    tau_duality_matrix = matrix_multiply(J_T, [[wrench[i]] for i in range(6)])  # 6x1
    # Flatten the result to a list
    tau_duality = [row[0] for row in tau_duality_matrix]
    print("Joint Torques from Kineostatic Duality:", tau_duality)  # Print torques

    print("\n----- Extra credit Trajectory Calculations and Simulation -----\n")

    # **Define initial and final joint angles in degrees for additional simulation**
    # These can be different from the main initial and final angles if desired
    additional_initial = [-50, -10, -110, -50, -50, 50]
    additional_final = [-10, -10, -10, -50, -50, 50]

    # **Convert joint angles to radians for simulation**
    additional_initial_joint_angles = [math.radians(angle) for angle in additional_initial]
    additional_final_joint_angles = [math.radians(angle) for angle in additional_final]

    # **Display initial and final joint angles**
    print("Additional Simulation - Initial Joint Angles (degrees):", additional_initial)
    print("Additional Simulation - Final Joint Angles (degrees):", additional_final)
    print("\n")

    # **Define trajectory duration and time parameters**
    tf_additional = 5    # Final time in seconds
    t0_additional = 0    # Initial time
    num_points_additional = 100

    # **Generate time vector manually**
    def linspace(start, stop, num):
        """
        Generate a list of evenly spaced values between start and stop.

        Parameters:
        - start (float): Starting value.
        - stop (float): Ending value.
        - num (int): Number of points.

        Returns:
        - list: List of evenly spaced values.
        """
        if num == 1:
            return [stop]
        step = (stop - start) / (num - 1)
        return [start + step * i for i in range(num)]

    t_additional = linspace(t0_additional, tf_additional, num_points_additional)

    # **Initialize joint trajectory arrays for different methods**
    def initialize_trajectory(num_points, num_joints):
        """
        Initialize a trajectory matrix filled with zeros.

        Parameters:
        - num_points (int): Number of points in the trajectory.
        - num_joints (int): Number of joints.

        Returns:
        - list of lists: Initialized trajectory matrix.
        """
        return [[0 for _ in range(num_joints)] for _ in range(num_points)]

    joint_trajectory_linear = initialize_trajectory(num_points_additional, len(additional_initial))
    joint_trajectory_cubic = initialize_trajectory(num_points_additional, len(additional_initial))
    joint_trajectory_quintic = initialize_trajectory(num_points_additional, len(additional_initial))
    joint_trajectory_bspline = initialize_trajectory(num_points_additional, len(additional_initial))
    joint_trajectory_bezier = initialize_trajectory(num_points_additional, len(additional_initial))
    joint_trajectory_trapezoidal = initialize_trajectory(num_points_additional, len(additional_initial))

    # **Initialize lists to store maximum accelerations for each joint (optional)**
    max_acc_cubic = []
    max_acc_quintic = []
    max_acc_trapezoidal = []

    # -------------------- Linear Interpolation --------------------
    print("----- Linear Interpolation Calculations -----")
    for i in range(len(additional_initial)):
        q0 = additional_initial[i]
        qf = additional_final[i]
        slope = (qf - q0) / tf_additional
        intercept = q0
        print(f"Joint {i+1}: q(t) = {intercept:.2f} + {slope:.2f} * t")
        
        # Linear interpolation: q(t) = q0 + (qf - q0) * (t / tf)
        for point in range(num_points_additional):
            joint_trajectory_linear[point][i] = q0 + (qf - q0) * (t_additional[point] / tf_additional)
    print("\n")

    # -------------------- Cubic Polynomial --------------------
    print("----- Cubic Polynomial Calculations -----")
    for i in range(len(additional_initial)):
        q0 = additional_initial[i]
        qf = additional_final[i]
        a0 = q0
        a1 = 0
        a2 = 3 * (qf - q0) / (tf_additional**2)
        a3 = -2 * (qf - q0) / (tf_additional**3)
        
        # Calculate and store maximum acceleration for each joint
        q_ddot_tf = 2*a2 + 6*a3*tf_additional
        max_acc = abs(q_ddot_tf)
        max_acc_cubic.append(max_acc)
        
        print(f"Joint {i+1}:")
        print(f"  q(t) = {a0:.2f} + {a1:.2f} * t + {a2:.2f} * t^2 + {a3:.2f} * t^3")
        print(f"  Maximum Acceleration at t={tf_additional}s: {max_acc:.2f} deg/s²")
        
        # Evaluate cubic polynomial for each time point: q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        for point in range(num_points_additional):
            joint_trajectory_cubic[point][i] = a0 + a1 * t_additional[point] + a2 * (t_additional[point] **2) + a3 * (t_additional[point] **3)
    print("\n")

    # -------------------- Quintic Polynomial --------------------
    print("----- Quintic Polynomial Calculations -----")
    def compute_quintic_coefficients(q0, qf, tf):
        # Boundary conditions: q0, qf, v0=0, vf=0, a0=0, af=0
        A = [
            [1, 0,      0,       0,        0,         0],
            [0, 1,      0,       0,        0,         0],
            [0, 0,      2,       0,        0,         0],
            [1, tf, tf**2, tf**3, tf**4, tf**5],
            [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
            [0, 0,      2,    6*tf,   12*tf**2, 20*tf**3]
        ]
        B = [q0, 0, 0, qf, 0, 0]
        # Solve the linear system A * coeffs = B
        # First, invert A
        try:
            A_inv = matrix_inverse(A)
        except ValueError:
            print("Matrix A is singular, cannot compute quintic coefficients.")
            return [0]*6
        # Multiply A_inv with B
        coeffs_matrix = matrix_multiply(A_inv, [[b] for b in B])  # 6x1 matrix
        coeffs = [row[0] for row in coeffs_matrix]
        return coeffs

    for i in range(len(additional_initial)):
        q0 = additional_initial[i]
        qf = additional_final[i]
        coeffs = compute_quintic_coefficients(q0, qf, tf_additional)
        
        # Calculate acceleration coefficients
        a0_coeff = coeffs[0]
        a1_coeff = coeffs[1]
        a2_coeff = coeffs[2]
        a3_coeff = coeffs[3]
        a4_coeff = coeffs[4]
        a5_coeff = coeffs[5]
        q_double_dot_tf = 2*a2_coeff + 6*a3_coeff*tf_additional + 12*a4_coeff*(tf_additional**2) + 20*a5_coeff*(tf_additional**3)
        max_acc_quintic.append(abs(q_double_dot_tf))
        
        print(f"Joint {i+1}:")
        print(f"  q(t) = {a0_coeff:.2f} + {a1_coeff:.2f} * t + {a2_coeff:.2f} * t^2 + {a3_coeff:.2f} * t^3 + {a4_coeff:.2f} * t^4 + {a5_coeff:.2f} * t^5")
        print(f"  Maximum Acceleration at t={tf_additional}s: {abs(q_double_dot_tf):.2f} deg/s²")
        
        # Evaluate quintic polynomial for each time point
        for point in range(num_points_additional):
            joint_trajectory_quintic[point][i] = (
                coeffs[0] +
                coeffs[1] * t_additional[point] +
                coeffs[2] * (t_additional[point] **2) +
                coeffs[3] * (t_additional[point] **3) +
                coeffs[4] * (t_additional[point] **4) +
                coeffs[5] * (t_additional[point] **5)
            )
    print("\n")

    # -------------------- B-Spline (Quadratic) --------------------
    print("----- B-Spline (Quadratic) Calculations -----")
    for i in range(len(additional_initial)):
        # Define control points: start, mid, end
        q0 = additional_initial[i]
        qf = additional_final[i]
        qm = (q0 + qf) / 2  # Midpoint
        
        # Knot vector for quadratic B-Spline with clamped ends
        knots = [0, 0, 0, 1, 1, 1]
        
        # B-Spline basis functions for k=2
        def basis(k, j, t_val, knots):
            if k == 0:
                return 1.0 if knots[j] <= t_val < knots[j+1] else 0.0
            else:
                denom1 = knots[j+k] - knots[j]
                denom2 = knots[j+k+1] - knots[j+1]
                term1 = 0.0
                term2 = 0.0
                if denom1 != 0:
                    term1 = ((t_val - knots[j]) / denom1) * basis(k-1, j, t_val, knots)
                if denom2 != 0:
                    term2 = ((knots[j+k+1] - t_val) / denom2) * basis(k-1, j+1, t_val, knots)
                return term1 + term2
        
        # Evaluate B-Spline at each time point
        spline_values = [0 for _ in range(num_points_additional)]
        for idx, ti in enumerate(t_additional):
            # Normalize ti to [0,1] for knot vector
            normalized_t = (ti - t0_additional) / (tf_additional - t0_additional)
            spline = 0.0
            for j in range(3):
                spline += [q0, qm, qf][j] * basis(2, j, normalized_t, knots)
            spline_values[idx] = spline
        # Correct assignment without using numpy
        for point in range(num_points_additional):
            joint_trajectory_bspline[point][i] = spline_values[point]
        
        print(f"Joint {i+1}:")
        print(f"  Control Points: P0 = {q0}, P1 = {qm}, P2 = {qf}")
        print(f"  B-Spline Equation: q(t) = P0 * B0,2(t) + P1 * B1,2(t) + P2 * B2,2(t)")
    print("\n")

    # -------------------- Bezier Curve (Linear) --------------------
    print("----- Bezier Curve (Linear) Calculations -----")
    for i in range(len(additional_initial)):
        q0 = additional_initial[i]
        qf = additional_final[i]
        
        # Linear Bezier: B(t) = (1 - t/tf)*q0 + (t/tf)*qf
        bezier_values = [0 for _ in range(num_points_additional)]
        for point in range(num_points_additional):
            t_ratio = t_additional[point] / tf_additional
            bezier_values[point] = (1 - t_ratio) * q0 + t_ratio * qf
        # Correct assignment without using numpy
        for point in range(num_points_additional):
            joint_trajectory_bezier[point][i] = bezier_values[point]
        
        print(f"Joint {i+1}:")
        print(f"  Control Points: P0 = {q0}, P1 = {qf}")
        print(f"  Bezier Curve Equation: B(t) = (1 - t/tf)*{q0} + (t/tf)*{qf}")
    print("\n")

    # -------------------- Trapezoidal Velocity Profile --------------------
    print("----- Trapezoidal Velocity Profile Calculations -----")
    # Define maximum velocity and acceleration for trapezoidal profile
    V_max = [10, 10, 10, 10, 10, 10]  # degrees per second
    A_max = [20, 20, 20, 20, 20, 20]  # degrees per second squared

    def trapezoidal_profile(q0, qf, V_max, A_max, tf, t):
        D = qf - q0
        direction = 1 if D >= 0 else -1
        D = abs(D)
        V_max = abs(V_max)
        A_max = abs(A_max)
        
        # Time to accelerate to V_max
        t_accel = V_max / A_max
        # Distance covered during acceleration
        D_accel = 0.5 * A_max * t_accel**2
        
        # Check if the profile is trapezoidal or triangular
        if 2 * D_accel > D:
            # Triangle profile (no cruise phase)
            t_accel = math.sqrt(D / A_max)
            V_peak = A_max * t_accel
            t_flat = 0
            print(f"  Triangle Profile: t_accel = {t_accel:.2f}s, V_peak = {V_peak:.2f} deg/s")
        else:
            # Trapezoidal profile
            D_flat = D - 2 * D_accel
            t_flat = D_flat / V_max
            V_peak = V_max
            print(f"  Trapezoidal Profile: t_accel = {t_accel:.2f}s, t_flat = {t_flat:.2f}s, V_peak = {V_peak:.2f} deg/s")
        
        T = 2 * t_accel + t_flat
        q = [0 for _ in range(len(t))]
        
        for idx, ti in enumerate(t):
            if ti < t_accel:
                # Acceleration phase
                q[idx] = 0.5 * A_max * ti**2
            elif ti < t_accel + t_flat:
                # Cruise phase
                q[idx] = D_accel + V_peak * (ti - t_accel)
            elif ti <= T:
                # Deceleration phase
                td = T - ti
                q[idx] = D - 0.5 * A_max * td**2
            else:
                q[idx] = D
        return [qi * direction for qi in q]

    for i in range(len(additional_initial)):
        q0 = additional_initial[i]
        qf = additional_final[i]
        print(f"Joint {i+1}:")
        trapezoidal_values = trapezoidal_profile(q0, qf, V_max[i], A_max[i], tf_additional, t_additional)
        for point in range(num_points_additional):
            joint_trajectory_trapezoidal[point][i] = trapezoidal_values[point]
        
        # Maximum acceleration is A_max
        print(f"  Maximum Acceleration: {A_max[i]} deg/s²\n")
    print("\n")

    # -------------------- Plotting All Trajectories --------------------
    plt.figure(figsize=(18, 12))

    methods = {
        'Linear Interpolation': joint_trajectory_linear,
        'Cubic Polynomial': joint_trajectory_cubic,
        'Quintic Polynomial': joint_trajectory_quintic,
        'B-Spline (Quadratic)': joint_trajectory_bspline,
        'Bezier Curve (Linear)': joint_trajectory_bezier,
        'Trapezoidal Velocity': joint_trajectory_trapezoidal
    }

    for idx, (method, trajectory) in enumerate(methods.items(), 1):
        plt.subplot(3, 2, idx)
        for j in range(len(trajectory[0])):
            # Extract the j-th joint's trajectory
            joint_traj = [trajectory[point][j] for point in range(num_points_additional)]
            plt.plot(t_additional, joint_traj, label=f'Joint {j+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Angles (degrees)')
        plt.title(f'Joint Trajectory using {method}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # -------------------- Dynamic Simulation on PyBullet --------------------
    # Function to get joint indices (assuming revolute joints and ignoring fixed joints)
    def get_revolute_joint_indices(robot):
        num_joints = p.getNumJoints(robot)
        revolute_joints = []
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot, i)
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE:
                revolute_joints.append(i)
        return revolute_joints

    # Connect to PyBullet for additional simulation
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # To load plane and other objects
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0,0,0])

    # Load the plane and robot
    plane_id = p.loadURDF("plane.urdf")
    # Replace 'your_robot.urdf' with the path to your robot's URDF file
    robot_urdf_path = r"C:\Users\anjar\OneDrive\Desktop\FOR\bullet3-master\examples\pybullet\gym\pybullet_data\xarm\xarm6_robot.urdf"
    try:
        robot_id = p.loadURDF(robot_urdf_path, useFixedBase=True)
    except Exception as e:
        print(f"Error: Unable to load URDF from path '{robot_urdf_path}'. Exception: {e}")
        p.disconnect()  # Disconnect if failed
        return

    # **Get revolute joint indices**
    revolute_joints = get_revolute_joint_indices(robot_id)
    print(f"Revolute Joint Indices: {revolute_joints}\n")

    # Function to set joint angles
    def set_joint_angles(robot, joint_indices, angles):
        for idx, joint in enumerate(joint_indices):
            p.setJointMotorControl2(bodyIndex=robot,
                                    jointIndex=joint,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=math.radians(angles[idx]),
                                    force=500)

    # Function to simulate a trajectory
    def simulate_trajectory(robot, joint_indices, trajectory, tf, num_points):
        slowdown_factor = 2  # Adjust this value to slow down the simulation
        for point in range(num_points):
            angles = trajectory[point]
            set_joint_angles(robot, joint_indices, angles)
            p.stepSimulation()
            time.sleep((tf / num_points) * slowdown_factor)  # Sync simulation with real time

    # -------------------- Simulate Each Trajectory Method --------------------
    print("----- Starting Dynamic Simulation -----\n")

    # List of trajectories to simulate
    trajectory_methods = {
        'Linear Interpolation': joint_trajectory_linear,
        'Cubic Polynomial': joint_trajectory_cubic,
        'Quintic Polynomial': joint_trajectory_quintic,
        'B-Spline (Quadratic)': joint_trajectory_bspline,
        'Bezier Curve (Linear)': joint_trajectory_bezier,
        'Trapezoidal Velocity': joint_trajectory_trapezoidal
    }

    for method_name, trajectory in trajectory_methods.items():
        print(f"Simulating {method_name}...")
        simulate_trajectory(robot_id, revolute_joints, trajectory, tf_additional, num_points_additional)
        print(f"{method_name} simulation completed.\n")
        time.sleep(1)  # Pause between methods

    # Disconnect from PyBullet
    p.disconnect()

if __name__ == "__main__":
    main()
